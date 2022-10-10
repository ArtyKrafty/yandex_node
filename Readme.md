## Baseline развертывания ноды в Yandex DataSphere


1. [Создание сервисного аккаунта](https://cloud.yandex.ru/docs/iam/operations/sa/create)
2. [Создание авторизованных ключей](https://cloud.yandex.ru/docs/iam/operations/authorized-key/create)
3. [Настроить окружение для развертывания самостоятельного сервиса](https://cloud.yandex.ru/docs/datasphere/operations/deploy/node-customization)
4. [API-ноды](https://cloud.yandex.ru/docs/datasphere/operations/deploy/node-api)
5. [IAM-token](https://cloud.yandex.ru/docs/iam/operations/iam-token/create-for-sa) для запроса в ноду

Содержание [видео по ноде](https://drive.google.com/file/d/1c29Q0ERSof-8Rxqa_cEmah7W9pcB9Okj/view?usp=sharing)

00:00:00 - Вступление. 
00:01:51 - Модель и постановка задачи. 
00:05:53 - Настройка окружения и подготовка requirments.txt. 
00:15:25 - Тестовый инференс. 
00:19:54 - Переход в Датасферу и дашборд. 
00:27:25 - Настройка окружения Датасферы через докер     
00:33:02 - Реестр контейнеров Yandex Cloud и создание сервисного аккаунта. 
00:34:44 - Авторизированные ключи. 
00:40:33 - Api ноды датасферы. 
00:51:05 - Активация окружения докера в ДатаСфере и отправка контейнера в реестр. 
00:59:21 - Создание ноды. 
01:04:12 - Запрос IaM токена. 
01:08:33 - Запрос в ноду. 
01:10:28 - Заключение. 



```python
import requests
import json
```
1. Создание модели

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import pandas as pd
from typing import List

# https://huggingface.co/cointegrated/rut5-base-absum 

def summarize(tokenizer, model,
    text, n_words=None, compression=None,
    max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, 
    **kwargs
):
    """
    Summarize the text
    The following parameters are mutually exclusive:
    - n_words (int) is an approximate number of words to generate.
    - compression (float) is an approximate length ratio of summary and original text.
    """
    if n_words:
        text = '[{}] '.format(n_words) + text
    elif compression:
        text = '[{0:.1g}] '.format(compression) + text
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **x, 
            max_length=max_length, num_beams=num_beams, 
            do_sample=do_sample, repetition_penalty=repetition_penalty, 
            **kwargs
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def summarize_news(input: List[str]) -> List:
    '''суммаризация'''
    MODEL_NAME = 'cointegrated/rut5-base-absum'
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    preds = []
    news = pd.read_csv(input[0])["inputs"].tolist()
    for rec in news:
        summary = summarize(tokenizer, model, rec)
        preds.append(summary)

    return preds
```


```python
URL = ["https://drive.google.com/uc?id=1ki7iNkXGdQ7lSmnze5Ox9CPzHWvvtCQ_"]

df = pd.read_csv(URL[0])
df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Мероприятие началось с одноименной песенки о д...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Амурский тигр и дальневосточный леопард практи...</td>
    </tr>
  </tbody>
</table>
</div>

2. Инференс модели


```python
summarize_news(URL)
```




    ['Встреча началась с песенки о дружбе. Участники встречи отвечали на вопросы викторины, рассказывали пословицы о дружбе.',
     'Амурский тигр и дальневосточный леопард спасены от угрозы полного исчезновения.']

3. Api ноды

```python

curl https://datasphere.api.cloud.yandex.net/datasphere/v1/nodes/<node_id>:execute \
   -X POST \
   -H "Authorization: Bearer <iam_token>"
   -d '{
      "folder_id": "<folder_id>",
      "node_id": "<node_id>",
      "input": { <input_variables> }
   }'

```


```python
token = ""
TOKEN = f"{token}"
NODE_ID = ""
FOLDER_ID = ""
BASE_URL = f"https://datasphere.api.cloud.yandex.net/datasphere/v1/nodes/{NODE_ID}:execute"
body = {

        "folderId": f"{FOLDER_ID}",
        "input": {"inputs": URL}


}

headers = {

        "Authorization": "Bearer" + TOKEN,
        "Content-Type": "appication/json"

}
```


```python
req = requests.post(BASE_URL, data=json.dumps(body), headers=headers)
```


```python
req
```




    <Response [200]>


4. Результат запроса в ноду

```python
req.json()["output"]["outputs"]
```




    ['Встреча началась с песенки о дружбе. Участники встречи отвечали на вопросы викторины, рассказывали пословицы о дружбе.',
     'Амурский тигр и дальневосточный леопард спасены от угрозы полного исчезновения.']

5. Bonus

Пример bash-скрипта для создания токена (чтобы добавить разрешение на его работу сначала `chmod +x ./my_script.sh`)

```bash
#!/bin/bash

OUTPUT=$(yc iam create-token)
echo "${OUTPUT}"

```
