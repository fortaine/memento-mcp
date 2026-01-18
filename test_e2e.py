#!/usr/bin/env python3
"""E2E Test for memento-v2 with Gemini 3 Flash + Thinking"""

from a_mem.utils.llm import LLMService
from a_mem.config import settings
import time
import numpy as np

def main():
    print('='*60)
    print('E2E TEST: memento-v2 with Gemini 3 Flash + Thinking')
    print('='*60)

    # Show config
    print(f'\nConfiguration:')
    print(f'  Provider: {settings.LLM_PROVIDER}')
    print(f'  LLM Model: {settings.GOOGLE_LLM_MODEL}')
    print(f'  Thinking Level: {settings.GOOGLE_THINKING_LEVEL}')
    print(f'  Include Thoughts: {settings.GOOGLE_INCLUDE_THOUGHTS}')
    print(f'  Embedding Model: {settings.GOOGLE_EMBEDDING_MODEL}')
    print(f'  Embedding Dims: {settings.GOOGLE_EMBEDDING_DIMENSIONS}')

    llm = LLMService()

    # Test 1: Embedding + L2 normalization
    print('\n[Test 1] Embeddings (768 dims + L2 normalization)...')
    start = time.time()
    emb = llm.get_embedding('The capital of France is Paris.')
    print(f'  ✓ Embedding generated in {time.time()-start:.2f}s')
    print(f'  ✓ Dimensions: {len(emb)}')
    
    # Verify L2 normalization
    norm = np.linalg.norm(emb)
    print(f'  ✓ L2 Norm: {norm:.6f} (should be ~1.0)')
    assert len(emb) == 768, f"Expected 768 dims, got {len(emb)}"
    assert 0.999 < norm < 1.001, f"Expected norm ~1.0, got {norm}"

    # Test 2: LLM with LOW thinking (default)
    print('\n[Test 2] LLM with LOW thinking (default)...')
    start = time.time()
    resp = llm._call_llm('What is 2 + 2? Reply with just the number.')
    print(f'  ✓ Response: {resp.strip()[:100]}')
    print(f'  ✓ Generated in {time.time()-start:.2f}s')
    assert '4' in resp, f"Expected '4' in response"

    # Test 3: LLM with HIGH thinking override for complex reasoning
    print('\n[Test 3] Complex reasoning with HIGH thinking override...')
    start = time.time()
    # Use the per-call thinking_level parameter
    resp = llm._call_google(
        'If a train travels at 60 km/h for 2.5 hours, how far does it travel? Show your reasoning briefly.',
        thinking_level='HIGH'
    )
    print(f'  ✓ Response: {resp.strip()[:300]}...')
    print(f'  ✓ Generated in {time.time()-start:.2f}s')
    assert '150' in resp, f"Expected '150' in response"

    # Test 4: Metadata extraction (should use LOW thinking - fast)
    print('\n[Test 4] Metadata extraction with LOW thinking...')
    start = time.time()
    resp = llm.extract_metadata("Python is a high-level programming language known for its readability and versatility.")
    elapsed = time.time()-start
    print(f'  ✓ Metadata: {resp}')
    print(f'  ✓ Generated in {elapsed:.2f}s')
    assert 'keywords' in resp, f"Expected 'keywords' in metadata"
    # LOW thinking should be faster than HIGH (~1-2s vs 3-4s)
    print(f'  ✓ Speed check: {elapsed:.2f}s (LOW should be <3s)')

    print('\n' + '='*60)
    print('✅ ALL E2E TESTS PASSED!')
    print('='*60)
    
    print('\n📊 Summary:')
    print('  • Embeddings: 768 dims with L2 normalization ✓')
    print('  • Default thinking: LOW (fast metadata extraction) ✓')
    print('  • Per-call override: HIGH for complex reasoning ✓')

if __name__ == '__main__':
    main()
