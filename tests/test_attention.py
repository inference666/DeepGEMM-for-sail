import random
import torch

import deep_gemm
from utils import test_mqa_logits, test_paged_mqa_logits, test_sparse_mqa_logits, set_acc_check, parse_deepgemm_string_re
from utils import is_ppu1v5_device, get_arch_version

def test_ks_ke(check_accuracy=True):
    if is_ppu1v5_device():
        print('Skipping INT8 numeric test: M890 does not support INT8 indexer')
        return
    # Single INT8 case: seq_len_q=1 with ks != 0, manually constructed (not via test_mqa_logits).
    # Verifies the kernel correctly offsets into KV when the valid window does not start at 0.
    # Only q[0, 0, 0]=1 and k[4, 0]=5 are non-zero, so logits[0, 4] = relu(1*5) * 1 * 1 = 5.
    print('Testing INT8 MQA Logits (seq_len_q=1, ks!=0):')
    sq, nh, hd = 1, 32, 128
    skv = 6
    q = torch.zeros((sq, nh, hd), device='cuda', dtype=torch.int8)
    k = torch.zeros((skv, hd), device='cuda', dtype=torch.int8)
    k_scale = torch.ones(skv, device='cuda', dtype=torch.float32)
    weights = torch.ones((sq, nh), device='cuda', dtype=torch.float32)
    q[0, 0, 0] = 1
    k[4, 0] = 5
    # ks != 0: only KV token 4 is within [ks, ke)
    ks = torch.tensor([4], device='cuda', dtype=torch.int32)
    ke = torch.tensor([5], device='cuda', dtype=torch.int32)
    # Reference: logits = (relu(q·k) * weights).sum(heads) * k_scale
    scores = torch.matmul(q.float(), k.float().T)            # [1, 32, 6]
    ref = (scores.relu() * weights[:, :, None]).sum(dim=1)  # [1, 6]
    ref = ref * k_scale[None, :]
    actual = deep_gemm.int8_mqa_logits(q, (k, k_scale), weights, ks, ke, clean_logits=False)
    if not check_accuracy:
        return
    torch.testing.assert_close(actual[0, 4].float(), ref[0, 4].float(), rtol=1e-3, atol=1e-3)
    print(f'  INT8 ks!=0 case: expected[0,4]={ref[0, 4].item():.4f}, actual[0,4]={actual[0, 4].item():.4f}')
    print("Passed\n")


def test_per_layer_cache_view(check_accuracy=True):
    if is_ppu1v5_device():
        print('Skipping INT8 cache-view test: M890 does not support INT8 indexer')
        return
    num_kv_blocks, block_kv, num_heads, head_dim = 8, 64, 32, 128
    num_layers = 4
    max_context_len = num_kv_blocks * block_kv

    v = (torch.arange(max_context_len) % 7 + 1).float()
    s = 0.001 * (1 + torch.arange(max_context_len).float())
    k = torch.full((max_context_len, head_dim), 0).to(torch.int8)
    k[:] = v.to(torch.int8).unsqueeze(1)
    page = torch.cat([k.view(torch.uint8).reshape(num_kv_blocks, block_kv * head_dim),
                      torch.from_numpy(s.numpy().astype('<f4')).view(torch.uint8).reshape(num_kv_blocks, block_kv * 4)],
                     dim=1).cuda()

    q = torch.ones(1, 1, num_heads, head_dim, device='cuda').to(torch.int8)
    weights = torch.full((1, num_heads), 1.0 / num_heads, device='cuda')
    block_table = torch.arange(num_kv_blocks, dtype=torch.int32, device='cuda').unsqueeze(0)
    context_lens = torch.full((1, 1), max_context_len, dtype=torch.int32, device='cuda')
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, deep_gemm.get_num_sms(),
        metadata_extra=(1, num_heads, head_dim, torch.int8))

    def check(fused_kv_cache):
        logits = deep_gemm.int8_paged_mqa_logits(
            q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
            max_context_len, clean_logits=False)
        if check_accuracy:
            valid = int((((logits.float().cpu()[0] / (head_dim * v * s)) - 1).abs() < 0.005).sum())
            assert valid == max_context_len

    fused_kv_cache = torch.zeros(num_kv_blocks, block_kv * (head_dim + 4),
                                 dtype=torch.uint8, device='cuda')
    fused_kv_cache.copy_(page)
    check(fused_kv_cache.view(num_kv_blocks, block_kv, 1, head_dim + 4))

    fused_kv_cache = torch.zeros(num_kv_blocks, num_layers, block_kv * (head_dim + 4),
                                 dtype=torch.uint8, device='cuda')
    fused_kv_cache[:, 3, :] = page
    fused_kv_cache = fused_kv_cache[:, 3].view(num_kv_blocks, block_kv, 1, head_dim + 4)
    assert fused_kv_cache.view(num_kv_blocks, block_kv, head_dim + 4).storage_offset() \
        == 3 * block_kv * (head_dim + 4)
    check(fused_kv_cache)


def test_paged_int8_page_boundaries():
    if is_ppu1v5_device():
        print('Skipping INT8 numeric test: M890 does not support INT8 indexer')
        return
    # Reject page32 on 810E until actlize fixes cross-cube TSM addressing.
    # Token-distinct exact scores validate supported layouts without BF16 rounding noise.
    context_lens = torch.tensor([31, 32, 33, 64, 65, 127], device='cuda', dtype=torch.int32)[:, None]
    batch_size, head_dim, max_context_len = context_lens.size(0), 128, 128
    values = (torch.arange(max_context_len, device='cuda', dtype=torch.int32) % 17 + 1).to(torch.int8)
    k = values[:, None].expand(max_context_len, head_dim).contiguous()
    scales = torch.ones(max_context_len, device='cuda', dtype=torch.float32)
    valid = torch.arange(max_context_len, device='cuda')[None, :] < context_lens
    for num_heads in (32, 64):
        q = torch.ones(batch_size, 1, num_heads, head_dim, device='cuda', dtype=torch.int8)
        reference = values.float()[None, :].expand(batch_size, -1) * (num_heads * head_dim)
        for page_kv in (32, 64, 128):
            pages = max_context_len // page_kv
            permutation = torch.arange(pages - 1, -1, -1, device='cuda', dtype=torch.int64)
            page_bytes = torch.cat((k.view(torch.uint8).view(pages, page_kv * head_dim),
                                    scales.view(torch.uint8).view(pages, page_kv * 4)), dim=1)
            cache = page_bytes[permutation].view(pages, page_kv, 1, head_dim + 4)
            block_table = permutation.to(torch.int32)[None, :].expand(batch_size, -1).contiguous()
            if page_kv == 32 and torch.cuda.get_device_capability() == (8, 0):
                def expect_page32_rejection(call):
                    try:
                        call()
                    except RuntimeError as error:
                        assert 'actlize TSM cross-cube' in str(error), str(error)
                    else:
                        raise AssertionError('810E page32 must be rejected')
                expect_page32_rejection(lambda: deep_gemm.get_paged_mqa_logits_metadata(
                    context_lens, page_kv, deep_gemm.get_num_sms(),
                    metadata_extra=(1, num_heads, head_dim, torch.int8)))
                # A schedule from a supported page size must not bypass the compute guard.
                metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    context_lens, 64, deep_gemm.get_num_sms(),
                    metadata_extra=(1, num_heads, head_dim, torch.int8))
                for dtype in (torch.float32, torch.bfloat16):
                    weights = torch.ones(batch_size, num_heads, device='cuda', dtype=dtype)
                    expect_page32_rejection(lambda: deep_gemm.int8_paged_mqa_logits(
                        q, cache, weights, context_lens, block_table, metadata, max_context_len))
                continue
            metadata = deep_gemm.get_paged_mqa_logits_metadata(
                context_lens, page_kv, deep_gemm.get_num_sms(),
                metadata_extra=(1, num_heads, head_dim, torch.int8))
            for dtype in (torch.float32, torch.bfloat16):
                weights = torch.ones(batch_size, num_heads, device='cuda', dtype=dtype)
                logits = deep_gemm.int8_paged_mqa_logits(
                    q, cache, weights, context_lens, block_table, metadata,
                    max_context_len)
                assert logits.dtype == dtype
                torch.testing.assert_close(logits[valid].float(), reference[valid], rtol=0, atol=0)
    torch.cuda.synchronize()
    print('PASS: INT8 paged boundaries, permuted pages, H32/H64, FP32/BF16')


def test_mqa_logits_loop():
    print('Testing MQA Logits:')
    data_types = ['fp8', 'fp4'] if is_ppu1v5_device() else ['int8']
    for data_type in data_types:
        for seq_len in (2048, 4096):
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if data_type == 'fp4' and num_heads != 64:
                    continue
                for seq_len_kv in (4096, 8192, 16384, 32768, 65536, 131072):
                    do_check = seq_len_kv < 32768
                    case = f'MqaLogits,data_type:{data_type},seq_len_q:{seq_len},seq_len_kv:{seq_len_kv},num_heads:{num_heads},head_dim:{head_dim}'
                    set_acc_check(do_check)
                    test_mqa_logits(parse_deepgemm_string_re(case))
                    if do_check and data_type == 'int8':
                        test_mqa_logits(parse_deepgemm_string_re(case + ',weights_dtype:bf16'))
    cases = [
        # Compressed logits and optional BF16 INT8 output.
        'MqaLogits,data_type:int8,seq_len_q:4096,seq_len_kv:8192,num_heads:64,head_dim:128,compressed_logits:1',
        'MqaLogits,data_type:int8,seq_len_q:1024,seq_len_kv:4096,num_heads:32,head_dim:128,weights_dtype:bf16',
    ]
    for case in cases:
        if is_ppu1v5_device() and 'data_type:int8' in case:
            continue
        set_acc_check(True)
        test_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_paged_mqa_logits_loop():
    print('Testing Paged MQA Logits:')
    data_types = ['fp8', 'fp4'] if is_ppu1v5_device() else ['int8']
    for data_type in data_types:
        for batch_size, next_n in [(1, 1), (64, 1), (128, 1)]:
            for num_heads, head_dim in [(32, 128), (64, 128)]:
                if data_type == 'fp4' and num_heads != 64:
                    continue
                for avg_kv in (8192, 32768):
                    do_check = avg_kv < 32768
                    case = f'PagedMqaLogits,data_type:{data_type},batch_size:{batch_size},next_n:{next_n},avg_context_len:{avg_kv},num_heads:{num_heads},head_dim:{head_dim}'
                    set_acc_check(do_check)
                    test_paged_mqa_logits(parse_deepgemm_string_re(case))
                    if do_check and data_type == 'int8':
                        test_paged_mqa_logits(parse_deepgemm_string_re(case + ',weights_dtype:bf16'))
    cases = [
        # Empty contexts and optional BF16 INT8 logits.
        'PagedMqaLogits,data_type:int8,batch_size:4,next_n:1,num_heads:64,head_dim:128,distribution:[20,10,0,0],page_kv:64',
        'PagedMqaLogits,data_type:int8,batch_size:16,next_n:1,num_heads:32,head_dim:128,distribution:[4090,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]',
        'PagedMqaLogits,data_type:int8,batch_size:64,next_n:1,num_heads:64,head_dim:128,avg_context_len:8192,weights_dtype:bf16',
        'PagedMqaLogits,data_type:int8,batch_size:64,next_n:1,num_heads:64,head_dim:128,avg_context_len:8192',
    ]
    if is_ppu1v5_device():
        cases.append('PagedMqaLogits,data_type:fp4,batch_size:64,next_n:1,num_heads:64,head_dim:128,avg_context_len:8192')
    for case in cases:
        if is_ppu1v5_device() and 'data_type:int8' in case:
            continue
        set_acc_check(True)
        test_paged_mqa_logits(parse_deepgemm_string_re(case))
    if not is_ppu1v5_device():
        # CUDA graph replay with changing context lengths.
        case = 'PagedMqaLogits,data_type:int8,batch_size:8,next_n:1,num_heads:32,head_dim:128,pre_distribution:[4090,0,1,0,1],distribution:[4090,0,1,0,1,100,200,300]'
        set_acc_check(False)
        test_paged_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_mqa_avg_logits_loop():
    if not is_ppu1v5_device():
        print('Skipping avg MQA logits: requires PPU1.5')
        return
    print('Testing Avg MQA Logits:')
    # Prefill covers all q_scale modes, both output dtypes, and an odd Q tail.
    prefill_cases = [
        'MqaAvgLogits,data_type:fp8,seq_len_q:129,seq_len_kv:1024,num_heads:4,head_dim:128,q_scale:0,logits_dtype:fp32,compressed_logits:1',
        'MqaAvgLogits,data_type:fp8,seq_len_q:256,seq_len_kv:1024,num_heads:4,head_dim:128,q_scale:1,logits_dtype:bf16,compressed_logits:1',
        'MqaAvgLogits,data_type:fp8,seq_len_q:256,seq_len_kv:1024,num_heads:4,head_dim:128,q_scale:2,logits_dtype:fp32,compressed_logits:1',
    ]
    # FP4 AVG requires compressed output. Keep Q/K tails, packed SFA
    # groups, and active/fallback balancing. On 39 CUs, Q4096/KV4096 uses
    # G=2 with multiple groups, covering their handoff for every Q-scale mode.
    prefill_cases += [
        f'MqaAvgLogits,data_type:fp4,seq_len_q:{seq_len_q},seq_len_kv:{seq_len_kv},'
        f'num_heads:4,head_dim:128,q_scale:{q_scale},logits_dtype:bf16,compressed_logits:1'
        for seq_len_q, seq_len_kv, q_scale in (
            (129, 258, 0), (129, 1025, 1), (129, 1025, 2), (1025, 2048, 0),
            (4096, 4096, 0), (4096, 4096, 1), (4096, 4096, 2),
        )
    ]
    # Explicit FP32 output remains supported alongside the BF16 default.
    prefill_cases.append('MqaAvgLogits,data_type:fp4,seq_len_q:129,seq_len_kv:258,num_heads:4,head_dim:128,q_scale:0,logits_dtype:fp32,compressed_logits:1')
    # Paged decode covers the four-head tile, empty contexts, and multiple next tokens.
    paged_cases = [
        'PagedMqaAvgLogits,data_type:fp8,batch_size:4,next_n:1,num_heads:4,head_dim:128,distribution:[0,1,127,1024],logits_dtype:bf16,page_kv:32',
        'PagedMqaAvgLogits,data_type:fp8,batch_size:8,next_n:2,num_heads:4,head_dim:128,avg_context_len:1024,logits_dtype:fp32',
        'PagedMqaAvgLogits,data_type:fp8,batch_size:8,next_n:5,num_heads:4,head_dim:128,avg_context_len:1024,logits_dtype:bf16,page_kv:128',
    ]
    # FP4 BF16 cases cover both padded Q layouts, all page sizes, request
    # switches, and the last request's partial Q/KV tiles.
    request_switch_batch = 32 * deep_gemm.get_num_sms() + 1
    paged_cases += [
        f'PagedMqaAvgLogits,data_type:fp4,batch_size:{request_switch_batch},next_n:1,num_heads:4,head_dim:128,context_pattern:[17],page_kv:32,max_context_len:128,logits_dtype:bf16',
        'PagedMqaAvgLogits,data_type:fp4,batch_size:7,next_n:4,num_heads:4,head_dim:128,distribution:[513,1,65,0,63,131,64],page_kv:64,max_context_len:1024,logits_dtype:bf16',
        f'PagedMqaAvgLogits,data_type:fp4,batch_size:{request_switch_batch},next_n:5,num_heads:4,head_dim:128,context_pattern:[17,0,1,63,64,65,129,257,513],page_kv:64,max_context_len:1024,logits_dtype:bf16',
        'PagedMqaAvgLogits,data_type:fp4,batch_size:1,next_n:6,num_heads:4,head_dim:128,distribution:[129],page_kv:128,max_context_len:256,logits_dtype:bf16',
    ]
    paged_cases.append('PagedMqaAvgLogits,data_type:fp4,batch_size:1,next_n:1,num_heads:4,head_dim:128,distribution:[17],page_kv:32,max_context_len:128,logits_dtype:fp32')
    set_acc_check(True)
    for case in prefill_cases:
        print(case)
        test_mqa_logits(parse_deepgemm_string_re(case))
    for case in paged_cases:
        print(case)
        test_paged_mqa_logits(parse_deepgemm_string_re(case))
    # Call directly: the shared AVG helper always enables compressed output.
    q = (torch.zeros((1, 4, 64), device='cuda', dtype=torch.int8),
         torch.zeros((1, 4), device='cuda', dtype=torch.int32))
    kv = (torch.zeros((1, 64), device='cuda', dtype=torch.int8),
          torch.zeros((1,), device='cuda', dtype=torch.int32))
    ks = torch.zeros((1,), device='cuda', dtype=torch.int32)
    ke = torch.ones((1,), device='cuda', dtype=torch.int32)
    try:
        deep_gemm.fp8_fp4_mqa_avg_logits(q, kv, ks, ke, max_seqlen_k=0)
    except RuntimeError as exc:
        assert 'AVG requires max_seqlen_k > 0' in str(exc), str(exc)
    else:
        raise AssertionError('AVG must reject max_seqlen_k=0')
    print("Passed\n")


def test_sparse_mqa_logits_loop():
    if get_arch_version() == 80:
        cases = [
            # Full blocks, true sparse selections, unaligned starts and the odd final Q.
            'SparseMqaLogits,data_type:int8,seq_len_q:32,seq_len_kv:1024,sparse_block_kv:8,num_max_sparse_blocks:256,check_metadata:1',
            'SparseMqaLogits,data_type:int8,seq_len_q:9,seq_len_kv:8191,sparse_block_kv:16,num_max_sparse_blocks:128,use_unaligned_ks:1,check_metadata:1',
            'SparseMqaLogits,data_type:int8,seq_len_q:9,seq_len_kv:8191,sparse_block_kv:8,num_max_sparse_blocks:128,use_unaligned_ks:1,check_metadata:1',
            'SparseMqaLogits,data_type:int8,seq_len_q:1,seq_len_kv:1,sparse_block_kv:16,num_max_sparse_blocks:4,check_metadata:1',
            'SparseMqaLogits,data_type:int8,seq_len_q:2,seq_len_kv:0,sparse_block_kv:16,num_max_sparse_blocks:4,check_metadata:1',
        ]
        set_acc_check(True)
        for case in cases:
            test_sparse_mqa_logits(parse_deepgemm_string_re(case))
    if not is_ppu1v5_device():
        print('Skipping FP4 sparse MQA logits: requires PPU1.5')
        return
    print('Testing Sparse MQA Logits:')
    cases = [
        # Prefill: both block sizes, aligned/unaligned starts, odd Q tails, and empty selections.
        'SparseMqaLogits,data_type:fp4,seq_len_q:512,seq_len_kv:1024,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:256,use_unaligned_ks:0,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:512,seq_len_kv:1024,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:256,use_unaligned_ks:0,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:9,seq_len_kv:639,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:128,use_unaligned_ks:1,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:9,seq_len_kv:639,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:128,use_unaligned_ks:1,check_metadata:1',
        'SparseMqaLogits,data_type:fp4,seq_len_q:2,seq_len_kv:0,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:4,use_unaligned_ks:0,check_metadata:1',
        # Paged decode: request boundaries, multi-entry schedules, and partial splits.
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:313,seq_len_kv:1023,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:128,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:512,seq_len_kv:65536,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:128,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:1,seq_len_kv:1025,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:256,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:2,seq_len_kv:0,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:4,check_metadata:1',
        # Largest merge bucket with more schedule slots than metadata threads.
        'SparseMqaLogits,data_type:fp4,seq_len_q:9,seq_len_kv:65535,num_heads:32,head_dim:128,sparse_block_kv:8,num_max_sparse_blocks:4096,use_unaligned_ks:1,check_metadata:1',
        'PagedSparseMqaLogits,data_type:fp4,seq_len_q:9,seq_len_kv:65535,num_heads:32,head_dim:128,sparse_block_kv:16,num_max_sparse_blocks:4096,page_kv:128,check_metadata:1',
    ]
    set_acc_check(True)
    for case in cases:
        print(case)
        test_sparse_mqa_logits(parse_deepgemm_string_re(case))
    print("Passed\n")


def test_int8_sparse_edges(check_accuracy=True):
    """Explicit merge/slot cases plus workspace replay on two independent streams."""
    if get_arch_version() != 80:
        print('Skipping INT8 sparse edges: requires PPU1.0')
        return
    from utils import decode_sparse_metadata, ref_sparse_metadata, verify_sparse_metadata
    from utils import ref_int8_sparse_mqa_logits, bf16_fma_reference, get_sparse_num_slots

    if check_accuracy:
        # Single-round BF16 FMA differs from FP32-then-BF16 at these midpoints.
        a = torch.tensor([3, 3, -3, -3, 3, -3, 3], device='cuda', dtype=torch.bfloat16)
        b = torch.tensor([174, 174, 174, 174, 174, 174, 176], device='cuda', dtype=torch.bfloat16)
        c = torch.tensor([2**-16, -2**-16, 2**-16, -2**-16, 0, 0, -528],
                         device='cuda', dtype=torch.bfloat16)
        expected = torch.tensor([524, 520, -520, -524, 520, -520, 0],
                                device='cuda', dtype=torch.bfloat16)
        torch.testing.assert_close(bf16_fma_reference(a, b, c), expected, rtol=0, atol=0)
        torch.testing.assert_close(bf16_fma_reference(a, b / 16384, c / 16384),
                                   expected / 16384, rtol=0, atol=0)
        print('BF16 single-round FMA reference regression passed')

    class SparseCase:
        def __init__(self, block, unaligned, repeats=1):
            self.block, self.unaligned, self.repeats = block, unaligned, repeats
            self.rows, self.capacity = 9 * repeats, 16
            self.num_kv = 34 * block
            head_values = torch.tensor([-128, -3, 0, 2, 127] * 7, device='cuda', dtype=torch.int8)[:32]
            self.q = head_values[None, :, None].expand(self.rows, 32, 128).contiguous()
            self.q[0].zero_()
            self.kv = (torch.empty((self.num_kv, 128), dtype=torch.int8, device='cuda'),
                       torch.empty(self.num_kv, dtype=torch.float32, device='cuda'))
            self.weights = torch.tensor([1 / 32, -1 / 64, 0, 1 / 128] * 8,
                                        dtype=torch.bfloat16, device='cuda').expand(self.rows, 32).contiguous()
            self.weights[1].fill_(-1 / 32)  # Valid logits must retain a negative weighted sum.
            self.ks = torch.empty(self.rows, dtype=torch.int32, device='cuda')
            self.ke = torch.empty_like(self.ks)
            self.indices = torch.empty((self.rows, self.capacity), dtype=torch.int32, device='cuda')
            self.update(0)

        def update(self, phase):
            start = (phase % (self.block - 1) + 1) if self.unaligned else 0
            length = 32 * self.block + 1
            # Partial overlap, disjoint pairs, an identical contiguous pair, an
            # empty/short pair, and a final singleton Q selecting the last token.
            even, odd = list(range(0, 32, 2)), list(range(1, 32, 2))
            partial = sorted(([1] + even[1:]) if phase % 2 == 0 else (even[:-1] + [31]))
            short = self.block + 1 if phase % 2 else 0
            lengths = ([length] * 6 + [short, 0 if short else 1, length]) * self.repeats
            blocks = [even, partial, even, odd, list(range(16)), list(range(16)),
                      list(range((short + self.block - 1) // self.block)),
                      [] if short else [0], list(range(15)) + [32]] * self.repeats
            self.counts = [min(self.capacity, (n + self.block - 1) // self.block) for n in lengths]
            padded = [row + [0] * (self.capacity - len(row)) for row in blocks]
            self.ks.fill_(start)
            self.ke.copy_(torch.tensor([start + n for n in lengths], dtype=torch.int32))
            self.indices.copy_(torch.tensor(padded, dtype=torch.int32))
            values = ((torch.arange(self.num_kv, device='cuda') + phase) % 7 - 3) * 40
            values[0], values[1] = 127, -128
            self.kv[0].copy_(values.to(torch.int8)[:, None].expand(-1, 128))
            self.kv[1].copy_(torch.tensor([1 / 128, 1 / 64, 1 / 32, 1 / 16], device='cuda')
                             .repeat((self.num_kv + 3) // 4)[:self.num_kv])

        def run(self):
            metadata = deep_gemm.get_sparse_mqa_logits_metadata(
                self.ks, self.ke, self.num_kv, self.indices, torch.int8,
                self.block, use_unaligned_ks=self.unaligned)
            return metadata, deep_gemm.int8_sparse_mqa_logits(
                self.q, self.kv, self.weights, metadata, self.capacity,
                self.block, use_unaligned_ks=self.unaligned)

        def check(self, outputs):
            metadata, logits = outputs
            verify_sparse_metadata(decode_sparse_metadata(metadata, self.block),
                                   ref_sparse_metadata(self.indices, self.counts, self.ks, self.ke,
                                                       self.num_kv, self.block, self.capacity, self.unaligned),
                                   self.rows)
            tokens = (self.indices.long()[:, :, None] * self.block
                      + (self.ks % self.block)[:, None, None]
                      + torch.arange(self.block, device='cuda')).flatten(1)
            valid = (torch.arange(self.capacity * self.block, device='cuda')[None, :]
                     < torch.tensor(self.counts, device='cuda')[:, None] * self.block)
            valid &= tokens >= self.ks[:, None]
            valid &= tokens < self.ke[:, None]
            safe_tokens = tokens.clamp(max=self.num_kv - 1)
            expected = ref_int8_sparse_mqa_logits(self.q, self.kv[0][safe_tokens], self.kv[1][safe_tokens], self.weights)
            assert logits.shape == (self.rows, 512) and logits.dtype == torch.bfloat16
            torch.testing.assert_close(logits[:, :self.capacity * self.block][valid], expected[valid], rtol=0, atol=0)
            assert (logits[1, :self.capacity * self.block][valid[1]] < 0).any()

    cases = [SparseCase(8, False), SparseCase(16, True)]
    for case in [*cases, SparseCase(16, False)]:
        outputs = case.run()
        if check_accuracy:
            case.check(outputs)
    if not check_accuracy:
        return

    # The final selected block straddles the actual K allocation, not just a
    # request boundary inside a larger pool. Only its first token is valid.
    tail_case = SparseCase(16, True)
    tail_case.num_kv = int(tail_case.ke.max().item())
    tail_case.kv = tuple(tensor[:tail_case.num_kv].clone() for tensor in tail_case.kv)
    tail_case.check(tail_case.run())

    def check_partial_layout(block, starts, ends, selections, unaligned=False):
        from utils import gather_sparse_reference

        rows, num_kv = len(starts), max(ends)
        capacity = (max(map(len, selections)) + 3) // 4 * 4
        counts = [len(selected) for selected in selections]
        assert counts == [min(capacity, (end - start + block - 1) // block)
                          for start, end in zip(starts, ends)]
        ks = torch.tensor(starts, device='cuda', dtype=torch.int32)
        ke = torch.tensor(ends, device='cuda', dtype=torch.int32)
        indices = torch.tensor([selected + [0] * (capacity - len(selected))
                                for selected in selections], device='cuda', dtype=torch.int32)
        # Vary every channel as well as token/head indices: constant-channel K
        # would miss a wrong AIU swizzle/plane stride in a padded 128-row stage.
        channel = torch.arange(128, device='cuda')
        head = torch.arange(32, device='cuda')
        q = ((torch.arange(rows, device='cuda')[:, None, None] * 11
              + head[None, :, None] * 7 + channel[None, None, :] * 5) % 255 - 127).to(torch.int8)
        q[0, 0, 0], q[-1, -1, -1] = -128, 127
        k = ((torch.arange(num_kv, device='cuda')[:, None] * 13
              + channel[None, :] * 17) % 255 - 127).to(torch.int8)
        k[-1, 0] = -128
        scales = ((torch.arange(num_kv, device='cuda') % 7 + 1) / 128).float()
        weights = ((head % 5 - 2) / 64).to(torch.bfloat16).expand(rows, 32).contiguous()
        metadata = deep_gemm.get_sparse_mqa_logits_metadata(
            ks, ke, num_kv, indices, torch.int8, block, use_unaligned_ks=unaligned)
        logits = deep_gemm.int8_sparse_mqa_logits(
            q, (k, scales), weights, metadata, capacity, block, use_unaligned_ks=unaligned)
        # Decode only the initialized split/schedule prefix, not the unused
        # worst-case metadata allocation. Header word 3 is alignment padding.
        num_splits, num_waves, _ = metadata[:12].view(torch.int32).cpu().tolist()
        split_bytes = 16 + (128 // block) * 8
        live_bytes = 16 + num_splits * split_bytes + num_waves * get_sparse_num_slots() * 16
        assert 16 <= live_bytes <= metadata.numel()
        decoded = decode_sparse_metadata(metadata[:live_bytes], block)
        verify_sparse_metadata(decoded, ref_sparse_metadata(
            indices, counts, ks, ke, num_kv, block, capacity, unaligned), rows)
        dense = deep_gemm.int8_mqa_logits(
            q, (k, scales), weights, ks, ke, clean_logits=False,
            max_seqlen_k=num_kv)
        # Q4 bounds the gathered-K FP32 temporary: the new B16 ring8 case
        # would otherwise materialize 161 * 2112 * 128 * 4 = 174 MB at once.
        # Both dense gather and the exact BF16 reference still check every
        # valid position, including real physical tails and the odd final Q.
        checked_positions = 0
        for begin in range(0, rows, 4):
            end = min(begin + 4, rows)
            gathered, valid = gather_sparse_reference(
                dense[begin:end], indices[begin:end], counts[begin:end], ks[begin:end], ke[begin:end], block)
            tokens = (indices[begin:end].long()[:, :, None] * block
                      + (ks[begin:end] % block)[:, None, None]
                      + torch.arange(block, device='cuda')).flatten(1).clamp(max=num_kv - 1)
            expected = ref_int8_sparse_mqa_logits(
                q[begin:end], k[tokens], scales[tokens], weights[begin:end])
            assert logits.dtype == torch.bfloat16
            torch.testing.assert_close(gathered[valid], expected[valid], rtol=0, atol=0)
            torch.testing.assert_close(logits[begin:end, :capacity * block][valid], expected[valid], rtol=0, atol=0)
            checked_positions += int(valid.sum().item())
        expected_positions = sum(
            max(0, min(block, end - (selected * block + start % block)))
            for start, end, selected_row in zip(starts, ends, selections)
            for selected in selected_row)
        assert checked_positions == expected_positions > 0
        return decoded

    # Unaligned unions can satisfy the first/last span test without being
    # contiguous. An INT8 partial-AIU fast path must keep this on the fallback.
    false_contiguous = check_partial_layout(
        8, [1, 2], [65, 66], [[0, 1, 2, 7], [0, 1, 2, 3]], unaligned=True)
    assert false_contiguous['num_kv_splits'] == 1
    merged = false_contiguous['splits'][0]
    offsets = [entry[0] for entry in merged['blocks'][:merged['num_kv_blocks']]]
    assert offsets == [1, 2, 9, 10, 17, 18, 26, 57]
    assert offsets[-1] - offsets[0] == (len(offsets) - 1) * 8
    assert not merged['is_contiguous']

    # Real, unpadded aligned K allocations with 1- and 15-token tails. Each
    # long/short Q pair has 65 union blocks (5/9 splits). Repeat B8 per schedule
    # slot so at least one entry retains 5 splits and crosses both stage rings.
    # The odd singleton and last partial split leave whole KV/Q warps inactive;
    # the short paired Q still needs its original per-block slot masks.
    for block, tail in ((8, 1), (16, 15)):
        end = 67 * block + tail
        repeats = get_sparse_num_slots() if block == 8 else 1
        starts = [3 * block, 66 * block] * repeats + [67 * block]
        selections = [list(range(3, 68)), [66, 67]] * repeats + [[67]]
        decoded = check_partial_layout(block, starts, [end] * len(starts), selections)
        if block == 8:
            assert any(entry[1] - entry[0] >= 5 for entry in decoded['schedule'])

    # Ring8 metadata prefetch needs a schedule entry with more than 8 splits.
    # Every long Q selects 132 gapped blocks. Pairing it with two nearby tail
    # blocks gives 133 aligned / 134 unaligned union blocks: 9 B8 or 17 B16
    # splits. Repeating once per schedule slot guarantees at least one whole
    # long pair survives slot-window cuts, regardless of atomic reservation order.
    # The final Q is odd, starts inside the real last block and has no padded K.
    for block, unaligned, tail in ((8, False, 1), (16, False, 15),
                                   (8, True, 3), (16, True, 3)):
        repeats = get_sparse_num_slots()
        off0, off1 = (1, 2) if unaligned else (0, 0)
        end = 265 * block + tail
        starts = [3 * block + off0, 264 * block + off1] * repeats + [265 * block + off0]
        selections = [list(range(3, 266, 2)), [264, 265]] * repeats + [[265]]
        assert len(selections[0]) == 132
        decoded = check_partial_layout(block, starts, [end] * len(starts), selections, unaligned=unaligned)
        union_blocks = 134 if unaligned else 133
        pair_splits = (union_blocks + 128 // block - 1) // (128 // block)
        assert decoded['num_kv_splits'] == repeats * pair_splits + 1
        live_entries = [entry for entry in decoded['schedule'] if entry[0] < entry[1]]
        longest_entry = max(entry[1] - entry[0] for entry in live_entries)
        assert longest_entry >= pair_splits >= 9, (block, unaligned, longest_entry)
        print(f'Ring8 metadata wrap passed: B={block}, unaligned={unaligned}, '
              f'rows={len(starts)}, splits={decoded["num_kv_splits"]}, max_entry={longest_entry}')

    # Invalid host contracts must fail before a device kernel can read a bad
    # pointer/header. Keep these calls on the already-warmed, valid fixture.
    base = cases[0]
    metadata = base.run()[0]
    valid_args = dict(q=base.q, kv_s=base.kv, weights=base.weights, metadata=metadata,
                      num_max_sparse_blocks=base.capacity, sparse_block_kv=base.block)
    invalid_args = [
        ({'q': base.q.to(torch.bfloat16)}, 'q must be'),
        ({'q': base.q[0]}, 'q must have shape'),
        ({'q': base.q.cpu()}, 'q must be'),
        ({'kv_s': (base.kv[0][:, :127].contiguous(), base.kv[1])}, 'kv_s[0]'),
        ({'kv_s': (base.kv[0], base.kv[1].to(torch.bfloat16))}, 'kv_s[1]'),
        ({'kv_s': (base.kv[0], base.kv[1][:-1])}, 'kv_s[1]'),
        ({'weights': base.weights.float()}, 'weights must be'),
        ({'metadata': metadata.view(torch.int32)}, 'metadata must be'),
        ({'metadata': metadata.cpu()}, 'same device'),
        ({'metadata': metadata[:-1]}, 'metadata is smaller'),
        ({'num_max_sparse_blocks': 3}, 'num_max_sparse_blocks'),
        ({'sparse_block_kv': 4}, 'sparse_block_kv'),
    ]
    for changes, message in invalid_args:
        try:
            deep_gemm.int8_sparse_mqa_logits(**{**valid_args, **changes})
        except RuntimeError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f'INT8 sparse accepted invalid {next(iter(changes))}')

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    captured = []
    for case, stream in zip(cases, streams):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case.run()  # Warm up this stream's metadata workspace before capture.
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = case.run()
        captured.append((graph, outputs))
    # Grow the same stream's eager workspace while the smaller graph remains alive.
    with torch.cuda.stream(streams[0]):
        larger = SparseCase(8, False, repeats=33)
        larger_outputs = larger.run()
    streams[0].synchronize()
    larger.check(larger_outputs)
    del larger, larger_outputs
    for phase in range(1, 4):
        for case, stream, (graph, _) in zip(cases, streams, captured):
            with torch.cuda.stream(stream):
                case.update(phase)
                graph.replay()
        for stream in streams:
            stream.synchronize()
        for case, (_, outputs) in zip(cases, captured):
            case.check(outputs)
    print('Passed INT8 sparse contracts, merge/slot boundaries, two-stream graph replay and workspace growth')


def test_int8_paged_sparse_edges(check_accuracy=True):
    """Native SoA cache addressing, request-local pairing and replay regressions."""
    if get_arch_version() != 80:
        print('Skipping INT8 paged sparse edges: requires PPU1.0')
        return
    from utils import pack_int8_sparse_paged_cache, unpack_int8_sparse_paged_cache
    from utils import check_int8_paged_sparse_reference_chunks, check_int8_paged_sparse_nonpaged
    from utils import decode_sparse_metadata, ref_paged_sparse_metadata, verify_paged_sparse_metadata
    from utils import get_sparse_num_slots

    class PagedCase:
        def __init__(self, block, page, repeats=1, long=False, long_blocks=132, all_same_request=False):
            self.block, self.page, self.long = block, page, long
            self.repeats = repeats
            self.long_blocks, self.all_same_request = long_blocks, all_same_request
            self.rows, self.cap = (161, long_blocks) if long else (9 * repeats, 16)
            self.tail = 1 if block == 8 else 15
            max_len = (2 * long_blocks + 1) * block + self.tail if long else max(4 * page + 3, 32 * block + 3)
            self.max_pages = (max_len + page - 1) // page
            self.physical_pages = max(8, self.max_pages)
            channel, head = torch.arange(128, device='cuda'), torch.arange(32, device='cuda')
            self.q = ((torch.arange(self.rows, device='cuda')[:, None, None] * 11
                       + head[None, :, None] * 7 + channel[None, None, :] * 5) % 255 - 127).to(torch.int8).unsqueeze(1)
            self.q[0, 0, 0, 0], self.q[-1, 0, -1, -1] = -128, 127
            self.weights = ((head % 5 - 2) / 64).to(torch.bfloat16).expand(self.rows, 32).contiguous()
            self.weights[1].fill_(-1 / 32)
            if not long:
                self.q[5].zero_()
            k = torch.zeros((self.physical_pages, page, 128), device='cuda', dtype=torch.int8)
            scales = torch.ones((self.physical_pages, page), device='cuda', dtype=torch.float32)
            # A full extra 512B of poison makes stride!=payload even for page128.
            self.cache = pack_int8_sparse_paged_cache(k, scales, extra_page_padding=512)
            self.lens = torch.empty(self.rows, device='cuda', dtype=torch.int32)
            self.table = torch.empty((self.rows, self.max_pages), device='cuda', dtype=torch.int32)
            self.indices = torch.empty((self.rows, self.cap), device='cuda', dtype=torch.int32)
            self.requests = torch.empty(self.rows, device='cuda', dtype=torch.int32)
            self.update(0)

        def update(self, phase):
            b, p = self.block, self.page
            self.phase = phase
            # Change run boundaries at graph replay, including 0 -> 1 -> 0.
            # A request ID identifies a contiguous run, not a globally unique key.
            if self.all_same_request or (not self.long and phase % 3 == 2):
                self.request_values = [0] * self.rows
            elif self.long:
                self.request_values = [i // 2 for i in range(self.rows)]
            else:
                pattern = ([0, 0, 1, 0, 0, 0, 1, 1, 1] if phase % 3 == 1 else
                           [0, 1, 1, 1, 2, 2, 0, 0, 0])
                self.request_values = pattern * self.repeats
            self.requests.copy_(torch.tensor(self.request_values, dtype=torch.int32))
            if self.long:
                lens = ([128 * b] * 160 + [1] if phase % 2 else
                        [(2 * self.long_blocks + 1) * b + self.tail] * 160 + [1])
                selected = [list(range(128)) if phase % 2 else list(range(3, 2 * self.long_blocks + 2, 2))
                            for _ in range(160)] + [[0]]
            else:
                # Odd request groups [1,3,2,3], zero context, token/block/page
                # boundaries, one zero-Q row and a real last partial block.
                lens = [0, 1, 3 * p + 1 + phase % 3, 16 * b, 4 * p + 1 + phase % 3,
                        2 * b - 1, 2 * b, 32 * b + 3, 32 * b + 1] * self.repeats
                if phase % 3 == 1:
                    lens[0], lens[1] = 1, 0  # Paired Q0 nonempty / Q1 empty.
                selected = []
                for row, length in enumerate(lens):
                    available = (length + b - 1) // b
                    count = min(self.cap, available)
                    blocks = (list(range(count)) if available <= self.cap else
                              [i * (available - 1) // (count - 1) for i in range(count)])
                    selected.append(blocks)
            self.counts = [min(self.cap, (length + b - 1) // b) for length in lens]
            assert self.counts == list(map(len, selected))
            # Existing sparse API contract: valid logical IDs are unique and
            # strictly increasing. Repeated physical pages remain legal below.
            assert all(all(left < right for left, right in zip(blocks, blocks[1:])) for blocks in selected)
            assert all(all(0 <= index * b < length for index in blocks) for length, blocks in zip(lens, selected))
            tables = []
            pattern = [0, 2, 1, 3, 7, 4, 7, 5, 6]
            for request in self.request_values:
                mapping = [(pattern[j % len(pattern)] + request * 3 + phase) % self.physical_pages
                           for j in range(self.max_pages)]
                mapping[-1] = self.physical_pages - 1
                tables.append(mapping)
            self.lens.copy_(torch.tensor(lens, dtype=torch.int32))
            self.table.copy_(torch.tensor(tables, dtype=torch.int32))
            self.indices.copy_(torch.tensor([blocks + [0] * (self.cap - len(blocks)) for blocks in selected], dtype=torch.int32))
            self.expected_valid = sum(min(b, length - index * b) for length, blocks in zip(lens, selected) for index in blocks)
            k, scales = unpack_int8_sparse_paged_cache(self.cache)
            token = torch.arange(self.physical_pages * p, device='cuda').view(self.physical_pages, p)
            channel = torch.arange(128, device='cuda')
            k.copy_(((token[:, :, None] * 13 + channel[None, None, :] * 17 + phase * 19) % 255 - 127).to(torch.int8))
            k[-1, -1, 0] = -128
            scales.copy_(((token + phase) % 7 + 1).float() / 128)

        def run(self):
            metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                self.lens, self.table, self.requests, self.page, self.indices, torch.int8, self.block)
            logits = deep_gemm.int8_paged_sparse_mqa_logits(
                self.q, self.cache, self.weights, metadata, self.cap, self.block)
            return metadata, logits

        def check(self, outputs, secondary=False):
            metadata, logits = outputs
            assert logits.shape == (self.rows, (self.cap * self.block + 511) // 512 * 512)
            assert logits.dtype == torch.bfloat16
            splits, waves, unaligned = metadata[:12].view(torch.int32).cpu().tolist()
            live = 16 + splits * (16 + 128 // self.block * 8) + waves * get_sparse_num_slots() * 16
            assert 16 <= live <= metadata.numel() and unaligned == 0
            decoded = decode_sparse_metadata(metadata[:live], self.block)
            ref = ref_paged_sparse_metadata(self.indices, self.counts, self.lens, self.table,
                                            self.requests, self.page, self.block, self.cap)
            verify_paged_sparse_metadata(decoded, ref, self.rows, self.requests)
            checked = check_int8_paged_sparse_reference_chunks(self.q, self.cache, self.weights, logits,
                self.lens, self.table, self.indices, self.counts, self.block, chunk_size=4)
            assert checked == self.expected_valid
            if secondary:
                check_int8_paged_sparse_nonpaged(self.q, self.cache, self.weights, logits, self.lens,
                    self.table, self.requests, self.indices, self.counts, self.block, self.cap)
            if not self.long:
                if self.counts[1]:
                    assert (logits[1, :min(self.block, int(self.lens[1].item()))] < 0).any()
                assert logits[5, :2 * self.block - 1].count_nonzero().item() == 0
                if self.phase % 3 == 1 and not self.all_same_request:
                    paired_empty = [s for s in decoded['splits'] if s['q_token_base'] == 0]
                    unpaired = [s for s in decoded['splits'] if s['q_token_base'] == 2]
                    assert paired_empty and all(s['q1_slot_base'] != 0xffff for s in paired_empty)
                    assert unpaired and all(s['q1_slot_base'] == 0xffff for s in unpaired)
                    assert any(e[2:] == (0, 2) for e in decoded['schedule'])
                    assert any(e[2:] == (2, 1) for e in decoded['schedule'])
            return decoded

    # Minimal P==B, non-power-of-two pages, normal deployment page sizes,
    # whole-stage same-page AIU, page crossing and physical-page reuse.
    cases = [PagedCase(block, page) for block, page in
             ((8, 8), (16, 16), (8, 24), (8, 32), (16, 48), (8, 64), (8, 128), (16, 128))]
    for case in cases:
        outputs = case.run()
        if check_accuracy:
            decoded = case.check(outputs, secondary=True)
            assert any(s['is_contiguous'] and s['num_kv_blocks'] < 128 // case.block
                       for s in decoded['splits']), 'Missing partial same-page contiguous hint'
            if case.page in (24, 48):
                assert any(case.page // case.block < s['num_kv_blocks'] < 128 // case.block and
                           not s['is_contiguous'] for s in decoded['splits']), \
                    'Missing partial non-power-of-two-page crossing without hint'
            if case.block == 8 and case.page == 32:
                # [0,2,1,3] page order gives a 16-block tile whose first/last
                # span looks contiguous but whose middle physical blocks are not.
                found = False
                for split in decoded['splits']:
                    offsets = [x[0] for x in split['blocks'][:split['num_kv_blocks']]]
                    if len(offsets) == 16 and offsets[-1] - offsets[0] == 15:
                        if offsets != list(range(offsets[0], offsets[0] + 16)):
                            assert not split['is_contiguous'], 'False-contiguous physical endpoints received a hint'
                            found = True
                assert found, 'Missing false-contiguous physical-page fixture'
            print(f'Native paged layout passed: B={case.block}, page_kv={case.page}, '
                  f'rows={case.rows}, valid_positions={case.expected_valid}')
    for block, page, selected_blocks in ((8, 64, 132), (16, 128, 132), (16, 128, 256)):
        long_case = PagedCase(block, page, long=True, long_blocks=selected_blocks)
        outputs = long_case.run()
        if check_accuracy:
            decoded = long_case.check(outputs)
            per_pair = (selected_blocks + 128 // block - 1) // (128 // block)
            assert decoded['num_kv_splits'] == 80 * per_pair + 1
            entries = [e for e in decoded['schedule'] if e[0] < e[1]]
            assert max(e[1] - e[0] for e in entries) == per_pair > 8
            # At least one whole pair survives the split-window cuts, giving
            # actual single-entry 9/17/32-split metadata ring8 wrap coverage.
            print(f'Native paged ring8 wrap passed: B={block}, page_kv={page}, '
                  f'pair_splits={per_pair}, splits={decoded["num_kv_splits"]}, '
                  f'waves={decoded["num_waves"]}, max_entry={max(e[1] - e[0] for e in entries)}')
            long_case.update(1)
            steady = long_case.check(long_case.run())
            assert max(e[1] - e[0] for e in steady['schedule']) == block
            print(f'Native paged contiguous entry passed: B={block}, page_kv={page}, max_entry={block}')
    if not check_accuracy:
        return

    # Complete empty schedule, as distinct from one empty row in a nonempty CTA.
    empty = PagedCase(8, 8)
    empty.lens.zero_(); empty.counts = [0] * empty.rows; empty.expected_valid = 0
    empty_outputs = empty.run()
    decoded = decode_sparse_metadata(empty_outputs[0], 8)
    assert decoded['num_kv_splits'] == 0
    verify_paged_sparse_metadata(decoded, ref_paged_sparse_metadata(empty.indices, empty.counts, empty.lens,
        empty.table, empty.requests, empty.page, empty.block, empty.cap), empty.rows, empty.requests)
    assert check_int8_paged_sparse_reference_chunks(empty.q, empty.cache, empty.weights, empty_outputs[1],
        empty.lens, empty.table, empty.indices, empty.counts, 8, chunk_size=4) == 0

    base = cases[5]
    metadata = base.run()[0]
    valid = dict(q=base.q, kv_cache=base.cache, weights=base.weights, metadata=metadata,
                 num_max_sparse_blocks=base.cap, sparse_block_kv=base.block)
    unaligned_cache = torch.empty((base.physical_pages, base.page, 1, 132), device='cuda', dtype=torch.uint8)
    misaligned_cache = torch.empty(base.physical_pages * base.cache.stride(0) + 1, device='cuda', dtype=torch.uint8).as_strided(
        base.cache.shape, base.cache.stride(), storage_offset=1)
    noncontiguous_q = torch.empty((base.rows, 1, 32, 256), device='cuda', dtype=torch.int8)[..., ::2]
    invalid = [{'q': base.q.to(torch.bfloat16)}, {'q': base.q[:, 0]}, {'q': base.q.cpu()},
               {'q': noncontiguous_q}, {'kv_cache': base.cache.to(torch.int8)},
               {'kv_cache': base.cache[..., :128]}, {'kv_cache': base.cache.cpu()},
               {'kv_cache': unaligned_cache}, {'kv_cache': misaligned_cache}, {'weights': base.weights.float()},
               {'weights': base.weights[:, :31].contiguous()}, {'metadata': metadata.view(torch.int32)},
               {'metadata': metadata.cpu()}, {'metadata': metadata[:-1]},
               {'num_max_sparse_blocks': 3}, {'sparse_block_kv': 4}]
    for changes in invalid:
        try:
            deep_gemm.int8_paged_sparse_mqa_logits(**{**valid, **changes})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'INT8 paged sparse accepted invalid {next(iter(changes))}')
    meta_args = dict(context_lens=base.lens, block_table=base.table, indices=base.requests,
                     page_kv=base.page, sparse_kv_block_indices=base.indices,
                     qk_dtype=torch.int8, sparse_block_kv=base.block)
    invalid_meta = [{'context_lens': base.lens[:, None]}, {'context_lens': base.lens.cpu()},
                    {'context_lens': base.lens.float()}, {'block_table': base.table.long()},
                    {'block_table': base.table.flatten()}, {'block_table': base.table.cpu()},
                    {'block_table': base.table[:, :0].contiguous()},
                    {'indices': base.requests[:, None]}, {'indices': base.requests.long()},
                    {'indices': base.requests.cpu()}, {'page_kv': 0}, {'page_kv': 9},
                    {'sparse_kv_block_indices': base.indices[:, :3].contiguous()},
                    {'sparse_kv_block_indices': base.indices.long()}]
    for changes in invalid_meta:
        try:
            deep_gemm.get_paged_sparse_mqa_logits_metadata(**{**meta_args, **changes})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'Paged metadata accepted invalid {next(iter(changes))}')

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graph_cases, captured = [cases[5], cases[-1]], []
    for case, stream in zip(graph_cases, streams):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = case.run()
        captured.append((graph, outputs))
    # Keep both graphs alive while the first stream obtains a larger workspace.
    with torch.cuda.stream(streams[0]):
        # 657 Q exceed two rounds of the 320-CTA metadata grid.
        larger = PagedCase(8, 64, repeats=73, all_same_request=True)
        larger_outputs = larger.run()
    streams[0].synchronize()
    larger.check(larger_outputs)
    print(f'Native paged all-same request/multi-CTA claims passed: rows={larger.rows}')
    # The same page table is legal for different requests. A one-token ID=1
    # run between ID=0 runs puts boundaries at 31/32: initial claims at 63/64
    # exercise a mismatch in ballot lane 31 and a full matching 32-ID chunk.
    with torch.cuda.stream(streams[0]):
        larger.requests[31] = 1
        larger_outputs = larger.run()
    streams[0].synchronize()
    larger.check(larger_outputs)
    print(f'Native paged recurrent-ID scan boundary passed: rows={larger.rows}, runs=[31,1,{larger.rows - 32}]')
    del larger, larger_outputs
    for phase in range(1, 4):
        for case, stream, (graph, _) in zip(graph_cases, streams, captured):
            with torch.cuda.stream(stream):
                case.update(phase)
                graph.replay()
        for stream in streams:
            stream.synchronize()
        for case, (_, outputs) in zip(graph_cases, captured):
            case.check(outputs)
    print('Passed native INT8 paged sparse SoA, aliases, strict unique IDs, repeated request-ID runs, ring8 wrap and two-stream graph replay')


def test_fp4_paged_shared_metadata_edges():
    """Exact FP4 outputs and shared metadata, including actual long entries.

    Packed inputs have an analytic reference independent of either sparse or
    dense compute. Every selected valid output is checked; allocation padding
    and tokens past context_lens are intentionally excluded.
    """
    if not is_ppu1v5_device():
        print('Skipping FP4 shared paged metadata edges: requires PPU1.5')
        return
    from utils import decode_sparse_metadata, ref_paged_sparse_metadata, verify_paged_sparse_metadata
    from utils import get_sparse_num_slots

    class PagedCase:
        def __init__(self, block, lengths, requests, capacity, page=128):
            self.block, self.page, self.cap = block, page, capacity
            self.rows = len(lengths)
            assert len(requests) == self.rows and self.rows > 0
            assert block in (8, 16) and page % block == 0
            assert 0 < capacity <= 4096 and capacity % 4 == 0
            self.max_pages = max(1, (max(lengths) + page - 1) // page)
            self.num_pages = self.max_pages + 3
            # Code 2/4/5/6 means E2M1 value 1/2/3/4. E8M0 127 means scale 1.
            # Values vary across physical tokens and blocks, so wrong page,
            # split-stage, or output-slot addresses cannot all give the same result.
            physical = torch.arange(self.num_pages * page, dtype=torch.int64)
            value_index = (physical * 5 + physical // 7 + physical // block * 3) % 4
            codes = torch.tensor([2, 4, 5, 6], dtype=torch.uint8)[value_index]
            self.k_values = (value_index + 1).view(self.num_pages, page)
            payload = page * 68
            stride = (payload + 511) // 512 * 512 + 512
            storage = torch.full((self.num_pages, stride), 0xA5, dtype=torch.uint8, device='cuda')
            packed = (codes | (codes << 4)).view(-1, 1).expand(-1, 64).contiguous()
            storage[:, :page * 64].copy_(packed.view(self.num_pages, page * 64))
            storage[:, page * 64:payload].fill_(0x7F)
            self.cache = storage.as_strided((self.num_pages, page, 1, 68), (stride, 68, 68, 1))
            row = torch.arange(self.rows)
            self.q_values = 1 + row % 2
            q_codes = torch.where(row % 2 == 0, 0x22, 0x44).to(torch.int8)
            self.q = (
                q_codes[:, None, None, None].expand(-1, 1, 32, 64).contiguous().cuda(),
                torch.full((self.rows, 1, 32), 0x7F7F7F7F, dtype=torch.int32, device='cuda'),
            )
            self.weight_factors = torch.tensor([1.0, -1.0, 0.5, 0.0])[row % 4]
            self.weights = (self.weight_factors[:, None] / 32).expand(-1, 32).contiguous().to(
                device='cuda', dtype=torch.bfloat16)
            self.lens = torch.empty(self.rows, dtype=torch.int32, device='cuda')
            self.requests = torch.empty_like(self.lens)
            self.table = torch.empty((self.rows, self.max_pages), dtype=torch.int32, device='cuda')
            self.blocks = torch.empty((self.rows, self.cap), dtype=torch.int32, device='cuda')
            self.permutation = list(range(self.num_pages))
            random.Random(1234 + block + self.rows).shuffle(self.permutation)
            self.update(lengths, requests)

        def update(self, lengths, requests, phase=0):
            assert len(lengths) == len(requests) == self.rows
            assert all(0 <= length <= self.max_pages * self.page for length in lengths)
            # Equal IDs in separate runs may use different page tables. Each
            # contiguous request run, including its odd tail, shares one table.
            tables, selected, counts = [], [], []
            run = -1
            for index, (length, request) in enumerate(zip(lengths, requests)):
                if index == 0 or requests[index - 1] != request:
                    run += 1
                tables.append([self.permutation[(j + run * 3 + phase) % self.num_pages]
                               for j in range(self.max_pages)])
                available = (length + self.block - 1) // self.block
                count = min(available, self.cap)
                blocks = list(range(count))
                # Select the last available block as well, exercising context
                # tails and a sparse hole when the capacity is smaller.
                if available > count and count:
                    blocks[-1] = available - 1
                selected.append(blocks + [0] * (self.cap - count))
                counts.append(count)
            lens_cpu = torch.tensor(lengths, dtype=torch.int32)
            tables_cpu = torch.tensor(tables, dtype=torch.int32)
            blocks_cpu = torch.tensor(selected, dtype=torch.int32)
            self.lens.copy_(lens_cpu)
            self.requests.copy_(torch.tensor(requests, dtype=torch.int32))
            self.table.copy_(tables_cpu)
            self.blocks.copy_(blocks_cpu)
            self.counts = counts
            logical = (blocks_cpu.long()[:, :, None] * self.block + torch.arange(self.block)).flatten(1)
            page_ids = tables_cpu.long().gather(1, logical // self.page)
            kv_values = self.k_values[page_ids, logical % self.page]
            # All 128 channels and 32 heads have the same positive Q/K value.
            # Dot, BF16 weights, and reduction are exact for these small powers
            # of two and integer multiples; negative/zero weights remain visible.
            self.expected = (128 * kv_values * self.q_values[:, None] *
                             self.weight_factors[:, None]).to(torch.bfloat16)
            self.valid = (torch.arange(self.cap * self.block)[None, :] <
                          torch.tensor(counts)[:, None] * self.block) & (logical < lens_cpu[:, None])

        def run(self):
            metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                self.lens, self.table, self.requests, self.page, self.blocks, torch.int8, self.block)
            logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                self.q, self.cache, self.weights, metadata, self.cap, self.block)
            return metadata, logits

        def check(self, outputs):
            metadata, logits = outputs
            assert logits.shape == (self.rows, self.cap * self.block)
            assert logits.dtype == torch.bfloat16
            splits, waves, unaligned = metadata[:12].view(torch.int32).cpu().tolist()
            live = 16 + splits * (16 + 128 // self.block * 8) + waves * get_sparse_num_slots() * 16
            assert 16 <= live <= metadata.numel() and unaligned == 0
            decoded = decode_sparse_metadata(metadata[:live], self.block)
            ref = ref_paged_sparse_metadata(self.blocks, self.counts, self.lens, self.table,
                                            self.requests, self.page, self.block, self.cap)
            verify_paged_sparse_metadata(decoded, ref, self.rows, self.requests)
            actual = logits[:, :self.cap * self.block].cpu()
            torch.testing.assert_close(actual[self.valid], self.expected[self.valid], rtol=0, atol=0)
            return decoded

    num_slots = get_sparse_num_slots()
    for block in (8, 16):
        for entry_splits in (9, 17, 32):
            rows = 2 * num_slots
            case = PagedCase(block, [128 * entry_splits] * rows, [0] * rows,
                             entry_splits * (128 // block))
            decoded = case.check(case.run())
            entries = [entry for entry in decoded['schedule'] if entry[0] < entry[1]]
            assert decoded['num_kv_splits'] == num_slots * entry_splits
            assert decoded['num_waves'] == 1 and len(entries) == num_slots
            assert all(end - begin == entry_splits for begin, end, _, _ in entries)
            print(f'FP4 shared metadata long entry passed: B={block}, entry_splits={entry_splits}, '
                  f'active_slots={len(entries)}, valid_positions={int(case.valid.sum())}', flush=True)

    for block in (8, 16):
        # More than two full producer grids also exercises cached run reuse on
        # dynamically claimed work. Boundaries at 31/32 exercise ballot lane 31.
        rows = 2 * (4 * num_slots) + 1
        requests = [0] * 31 + [1] + [0] * (rows - 32)
        pattern = [0, 1, block - 1, block + 1, 127, 128, 129, 255, 257]
        lengths = [pattern[i % len(pattern)] for i in range(rows)]
        lengths[0], lengths[1], lengths[-1] = 129, 0, 129
        case = PagedCase(block, lengths, requests, 16)
        decoded = case.check(case.run())
        assert case.rows % 2 == 1
        assert any(s['is_contiguous'] and s['num_kv_blocks'] < 128 // block
                   for s in decoded['splits']), 'Missing partial same-page hint'
        assert any(not s['is_contiguous'] for s in decoded['splits']), 'Missing page crossing'
        paired_empty = [s for s in decoded['splits'] if s['q_token_base'] == 0]
        assert paired_empty and all(s['q1_slot_base'] != 0xFFFF for s in paired_empty)
        unpaired = [s for s in decoded['splits'] if s['q_token_base'] == rows - 1]
        assert unpaired and all(s['q1_slot_base'] == 0xFFFF for s in unpaired)
        print(f'FP4 shared metadata repeated-run/partial passed: B={block}, rows={rows}, '
              f'valid_positions={int(case.valid.sum())}', flush=True)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case.run()  # Allocate this stream's workspace before capture.
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = case.run()
        for phase in range(3):
            # Change Q pairing and page mappings while retaining input addresses.
            replay_requests = ([0] * rows if phase == 0 else
                               [0] * 32 + [1] + [0] * (rows - 33))
            replay_lengths = ([0] * rows if phase == 2 else lengths[phase:] + lengths[:phase])
            with torch.cuda.stream(stream):
                case.update(replay_lengths, replay_requests, phase + 1)
                graph.replay()
            stream.synchronize()
            decoded = case.check(outputs)
            if phase == 2:
                assert decoded['num_kv_splits'] == 0
                assert all(entry == (0, 0, 0, 0) for entry in decoded['schedule'])
        print(f'FP4 shared metadata graph replay passed: B={block}, changed runs/pages/lengths and empty schedule',
              flush=True)
    print('Passed FP4 shared paged metadata edges', flush=True)


def test_sparse_workspace():
    """Check sparse workspace lifetime across graph replay and larger eager calls.

    Packed FP4 inputs use an exact analytic logits reference. Metadata is also
    checked against the existing sparse schedule reference on two streams.
    """
    if not is_ppu1v5_device():
        print('Skipping sparse workspace: requires PPU1.5')
        return
    from utils import decode_sparse_metadata, ref_paged_sparse_metadata, verify_paged_sparse_metadata

    PAGE_KV = 128
    BLOCK_KV = 8
    TOPK_BLOCKS = 32
    NUM_PAGES = 32

    class PagedCase:
        def __init__(self, rows, phase=0):
            self.rows = rows
            self.q = (
                torch.full((rows, 1, 32, 64), 0x22, dtype=torch.int8, device="cuda"),
                torch.full((rows, 1, 32), 0x7F7F7F7F, dtype=torch.int32, device="cuda"),
            )
            self.weights = torch.full((rows, 32), 1 / 32, dtype=torch.bfloat16, device="cuda")
            # Fused page: all packed K payloads followed by all scale words.
            # E2M1 code 2 = 1, code 4 = 2; E8M0 byte 127 = scale 1.
            storage = torch.empty((NUM_PAGES, PAGE_KV * 68), dtype=torch.uint8, device="cuda")
            for page in range(NUM_PAGES):
                storage[page, : PAGE_KV * 64].fill_(0x22 if page % 2 == 0 else 0x44)
            storage[:, PAGE_KV * 64 :].fill_(0x7F)
            self.kv = storage.view(NUM_PAGES, PAGE_KV, 1, 68)
            self.lens = torch.empty(rows, dtype=torch.int32, device="cuda")
            self.table = torch.empty((rows, 9), dtype=torch.int32, device="cuda")
            self.blocks = torch.empty((rows, TOPK_BLOCKS), dtype=torch.int32, device="cuda")
            self.requests = torch.tensor([i // 3 for i in range(rows)], dtype=torch.int32, device="cuda")
            self.update(phase)

        def update(self, phase):
            boundaries = [127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025]
            lens = [boundaries[(i + phase) % len(boundaries)] for i in range(self.rows)]
            # Same request => identical table rows, including a pair crossing a page.
            tables = [[(j * 7 + (i // 3) * 3 + phase) % NUM_PAGES for j in range(9)]
                      for i in range(self.rows)]
            blocks, counts, expected, valid = [], [], [], []
            for length, table in zip(lens, tables):
                available = (length + BLOCK_KV - 1) // BLOCK_KV
                count = min(available, TOPK_BLOCKS)
                selected = list(range(count))
                if available > count:
                    selected[-1] = available - 1
                blocks.append(selected + [0] * (TOPK_BLOCKS - count))
                counts.append(count)
                row_expected, row_valid = [], []
                for slot in range(TOPK_BLOCKS):
                    block = blocks[-1][slot]
                    physical_page = table[block // (PAGE_KV // BLOCK_KV)]
                    for token in range(BLOCK_KV):
                        row_expected.append(128 * (1 + physical_page % 2))
                        row_valid.append(slot < count and block * BLOCK_KV + token < length)
                expected.append(row_expected)
                valid.append(row_valid)
            self.lens.copy_(torch.tensor(lens, dtype=torch.int32))
            self.table.copy_(torch.tensor(tables, dtype=torch.int32))
            self.blocks.copy_(torch.tensor(blocks, dtype=torch.int32))
            self.counts = counts
            self.expected = torch.tensor(expected, dtype=torch.bfloat16)
            self.valid = torch.tensor(valid, dtype=torch.bool)

        def run(self):
            metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                self.lens, self.table, self.requests, PAGE_KV,
                self.blocks, torch.int8, BLOCK_KV,
            )
            logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                self.q, self.kv, self.weights, metadata, TOPK_BLOCKS, BLOCK_KV,
            )
            return metadata, logits

        def check(self, outputs):
            metadata, logits = outputs
            ref = ref_paged_sparse_metadata(
                self.blocks, self.counts, self.lens, self.table, self.requests,
                PAGE_KV, BLOCK_KV, TOPK_BLOCKS,
            )
            verify_paged_sparse_metadata(
                decode_sparse_metadata(metadata, BLOCK_KV), ref, self.rows, self.requests,
            )
            actual = logits[:, : TOPK_BLOCKS * BLOCK_KV].cpu()
            torch.testing.assert_close(actual[self.valid], self.expected[self.valid], rtol=0, atol=0)


    def capture(case, stream):
        with torch.cuda.stream(stream):
            for _ in range(3):
                case.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = case.run()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        case.check(outputs)
        return graph, outputs

    print("device:", torch.cuda.get_device_name(), "DeepGemm:", deep_gemm.__file__, flush=True)
    torch.cuda.init()
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(first):
        small_a = PagedCase(9)
    with torch.cuda.stream(second):
        small_b = PagedCase(13, phase=3)
    first.synchronize()
    second.synchronize()
    graph_a, output_a = capture(small_a, first)
    graph_b, output_b = capture(small_b, second)
    print("PASS: capture + exact logits/metadata reference on two streams", flush=True)

    # 2048 query rows exceed either captured case's old workspace allocation.
    # Release the eager intermediates and keep allocator churn alive through replay.
    with torch.cuda.stream(first):
        large = PagedCase(2048, phase=5)
        large_outputs = large.run()
    first.synchronize()
    large.check(large_outputs)
    del large_outputs, large
    torch.cuda.synchronize()
    churn = [torch.full((384 + rows * 8,), 0xA5, dtype=torch.uint8, device="cuda")
             for rows in (9, 13, 512, 2048)]
    torch.cuda.synchronize()
    print("PASS: 2048-row eager call after small captures", flush=True)

    for phase in range(20):
        with torch.cuda.stream(first):
            small_a.update(phase)
            graph_a.replay()
        with torch.cuda.stream(second):
            small_b.update(phase + 3)
            graph_b.replay()
        first.synchronize()
        second.synchronize()
        small_a.check(output_a)
        small_b.check(output_b)
    assert len(churn) == 4
    print("PASS: 40 replays, changing lengths/page mappings, exact metadata + logits", flush=True)

    cold_stream = torch.cuda.Stream()
    with torch.cuda.stream(cold_stream):
        cold_case = PagedCase(3)
    cold_stream.synchronize()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=cold_stream):
            cold_case.run()
    except RuntimeError as error:
        assert "Warm up sparse MQA metadata" in str(error), str(error)
    else:
        raise AssertionError("First workspace allocation during capture must be rejected")
    graph_c, output_c = capture(cold_case, cold_stream)
    cold_case.check(output_c)
    print("PASS: cold capture rejected; warmup then capture/replay succeeds", flush=True)


def test_nvtx():
    print('Testing nvtx dump:')
    import torch.cuda.nvtx as nvtx
    data_type = 'fp8' if is_ppu1v5_device() else 'int8'
    nvtx.range_push("paged_mqa_logits")
    for batch_size in (1, 64):
        for num_heads in (32, 64):
            case = f'PagedMqaLogits,data_type:{data_type},batch_size:{batch_size},next_n:1,avg_context_len:8192,num_heads:{num_heads},head_dim:128'
            set_acc_check(False)
            test_paged_mqa_logits(parse_deepgemm_string_re(case))
    nvtx.range_pop()
    print("Passed\n")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    # test_nvtx()

    test_ks_ke()
    test_per_layer_cache_view()
    test_paged_int8_page_boundaries()
    test_mqa_logits_loop()
    test_paged_mqa_logits_loop()
    test_mqa_avg_logits_loop()
    test_sparse_mqa_logits_loop()
    test_int8_sparse_edges()
    test_int8_paged_sparse_edges()
    test_fp4_paged_shared_metadata_edges()
    test_sparse_workspace()
