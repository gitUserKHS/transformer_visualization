# TinyStories Educational Subset

이 폴더의 `tinystories.jsonl`은
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
학습 split에서 5,500개 행을 가져온 교육용 고정 부분집합입니다.

- 원본 연구: *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- 원본 라이선스: CDLA-Sharing-1.0
- 추출 방식: Hugging Face Dataset Viewer API의 첫 5,500개 행을 100개 단위로
  요청한 뒤, 시드 42로 고정된 배치 순서로 저장
- 용도: 앞 5,000개 학습, 뒤 500개 검증

원본 데이터의 라이선스와 고지를 유지해야 합니다.

