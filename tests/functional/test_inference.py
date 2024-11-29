def test_inference_pipeline(inference_model, exp_dataset, inference_kwargs):
    # Run inference
    res = inference_model(exp_dataset[0], **inference_kwargs)

    # Verify results
    assert res is not None
    assert hasattr(res, "importance_sampling")
    assert hasattr(res.importance_sampling, "eff")
    assert 0 < res.importance_sampling.eff < 1.0
