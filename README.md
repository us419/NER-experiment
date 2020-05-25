# NER-experiment
* Bert와 Roberta를 pretrained model을 가져온 후 fine tunning 하여 Named entity recognition 수행 (English text)
* Bert로 82.1 f1 score, Roberta로 82.3 f1 score
* 이전에 glove embedding + BiLstm + crf 모델로 84.7 f1 score 나옴
* Bert 모델에 crf layer를 붙여보았으나 성능이 다소 떨어짐

## 원인 분석
* OOM 문제로 인해 Bert와 Roberta의 pretrained model을 가장 작은 base 모델로 가져옴
* Roberta의 pretrained model의 경우 text를 전부 소문자로 바꾸고 train하여서 NER에 적합하지 않음
* Bert 모델들은 전처리를 하지 않음 (BertTokenizer에 너무 의존함)
* crf layer를 TorchCRF 모듈에서 가져왔는데 공용서버의 환경문제 상 호환이 제대로 되지 않음

## further study
* 메모리 문제상 큰 사이즈의 모델을 사용하지 못하므로 albert를 사용해 볼 것 (distilbert는 pretrain model이 base밖에 없음)
* preprocessing 고민해 볼 것
* Bert를 word embedding으로만 사용하여 BiLSTM + crf 에 붙여볼 것 
