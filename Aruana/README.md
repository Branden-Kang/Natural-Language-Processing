# How to label text for sentiment analysis — good practices

```python 
from aruana import Aruana
aruana = Aruana('pt-br')
sentiment = aruana.random_classification(data['text'], classes=[0,1], balanced=True)
data['sentiment'] = sentiment

from aruana import Aruana
aruana = Aruana('pt-br')
texts = aruana.replace_with_blob(data['text'])
data['text'] = texts
```
