## Fine tuning KoGPT2 and deploy KoGPT2 based model in Amazon SageMaker



이 튜토리얼은 KoGPT2 모델을 활용해서 텍스트 감정 분석, 짝 문장 찾기과 같은 여러 NLP 문제를 해결하기 위한 fine-tuning 방법과 만들어진 모델을 Amazon SageMaker를 활용해서 배포하는 방법을 설명합니다.



**Tutorial** 1: Fine-tuning (Coming soon)

**Tutorial 2**: [Amazon SageMaker MXNet inference 컨테이너를 활용한 확장성 있는 KoGPT2 모델 배포하기](./sagemaker-deploy-ko.md)

- [Notebook to deploy a model](./deploy-sm-notebook.ipynb)
- [Inference code](./gpt2-inference.py)



This is a tutorial on how to do fine-tuning on pre-trained KoGPT2 model to solve downstream NLP tasks such as sentimental analysis and paraphrase detection, and how to deploy the NLP model to Amazon SageMaker for production purpose at scale.

**Tutorial** 1: Fine-tuning (Coming soon)

**Tutorial 2**: [Building scalable KoGPT2 model inference using Amazon SageMaker by extending MXNet inference container](./sagemaker-deploy-en.md)

- [Notebook to deploy a model](./deploy-sm-notebook.ipynb)
- [Inference code](./gpt2-inference.py)



**Resources**

- KoGPT2 https://github.com/SKT-AI/KoGPT2
- GluonNLP https://gluon-nlp.mxnet.io/
- SageMaker MXNet serving container git repository https://github.com/aws/sagemaker-mxnet-serving-container
- Amazon SageMaker Python SDK - Deploy Endpoints from Model Data https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#deploy-endpoints-from-model-data
- MXNet Model Serving sample https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_gluon_sentiment



## License

This library is licensed under the MIT-0 License. See the LICENSE file.