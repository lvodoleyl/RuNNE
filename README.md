# Соревнование RuNNE: извлечение именованных сущностей в few-shot режиме
## Введение
Извлечение именованных сущностей – одна из самых востребованных на практике задач извлечения информации – предполагает поиск в тексте упоминаний имен, организаций, топонимов и других сущностей.  Соревнование RuNNE посвящено задаче извлечения вложенных именованных сущностей. Разметка данных допускает следующие случаи: внутри одной именованной сущности находится другая именованная сущность. Так, например в сущность класса Organization “Московский драматический театр имени М. Н. Ермоловой” вложена сущность типа Person – “М. Н. Ермоловой”. 


## Данные
Соревнование проводится на материале корпуса NEREL [1], собранного из новостных текстов WikiNews на русском языке. В корпусе NEREL представлено 29 классов различных сущностей, а глубина вложенности сущностей достигает 6 уровней разметки. 

Данные предоставляются участникам в виде размеченных документов. Формат разметки – BRAT.


## Постановка задачи 
В рамках соревнования RuNNE мы предлагаем  участникам рассмотреть few shot  постановку задачи. 
Задача предполагает извлечение вложенных именованных сущностей,
В обучающем множестве большая часть типов именованных сущностей  встречается достаточно часто, а некоторое количество специально отобранных типов – встречается всего несколько раз, 
В тестовом множестве все типы сущностей представлены одинаково.

Таким образом, участникам предстоит разработать модели извлечения вложенных именованных сущностей, поддерживающие few-shot режим. 

## Оценка соревнования
В качестве метрики качества в соревновании RuNNE используется макро усреднение F1-меры в двух вариантах:  по классам известных сущностей (общая постановка задачи извлечения вложенных именованных сущностей) и по классам новых именованных сущностей (few-shot постановка). 



## Правила участия 
* Участникам соревнования разрешается использовать любые дополнительные материалы и любые предобученные модели, за исключением непосредственной разметки тестового множества. 
* Участники могут самостоятельно разметить дополнительные данные в соответствии с опубликованными инструкциями. При этом, организаторы соревнования будут просить участников опубликовать новые размеченные данные в открытом доступе. 

## Полезные ссылки
* Codalab: https://codalab.lisn.upsaclay.fr/competitions/1142
* Github: https://github.com/dialogue-evaluation/RuNNE
* Tlg: t.me/deval_RuNNE

## Ключевые даты 
* 29 декабря 2021 – публикация обучающих данных 
* 4 февраля 2022 –  публикация тестовых данных
* 14 февраля 2022 – закрытие тестирования 
* 15 марта – завершаем прием статей 


## Организаторы
* Наталья Лукашевич (МГУ)
* Екатерина Артемова (Huawei, НИУ ВШЭ)
* Татьяна Батура (НГУ, ИСИ СО РАН)
* Павел Браславский (НИУ ВШЭ, УРФУ)
* Владимир Иванов (Иннополис)
* Елена Тутубалина (Sber AI, НИУ ВШЭ)

1. Loukachevitch, Natalia, Ekaterina Artemova, Tatiana Batura, Pavel Braslavski, Ilia Denisov, Vladimir Ivanov, Suresh Manandhar, Alexander Pugachev, and Elena Tutubalina. "NEREL: A Russian Dataset with Nested Named Entities and Relations." In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021) (pp. 876-885). https://aclanthology.org/2021.ranlp-main.100.pdf




