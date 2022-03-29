# Соревнование RuNNE: извлечение именованных сущностей в few-shot режиме
## Решение команды Seldon

В рамках соревнования наша команда использовала модель Sodner - https://github.com/foxlf823/sodner. 

Были переразмечены по методологии создателей корпуса __NEREL__[1], а также добавлены дополнительные размеченные данные для few-short классов. Итоговый датасет представлен в __data.zip__ (cleared - переразмеченная train выборка, few-short - дополнительно размеченные данные).

Для работы с моделью Sodner использовались функции, представленные в файле __scripts.py__. 

Лучший результат показала модель с параметрами, представленными в __nerel_config.jsonnet__. 

По результатам соревнования была написана статья "Sodner for Russian nested named entity recognition" Kirill Abrosimov, Arina Mosyagina [FIXME: после публикации оформить по правилам и дать ссылку], в которой описываются используемые модели, результаты, а также работа с датасетом. 


[1] Loukachevitch, Natalia, Ekaterina Artemova, Tatiana Batura, Pavel Braslavski, Ilia Denisov, Vladimir Ivanov, Suresh Manandhar, Alexander Pugachev, and Elena Tutubalina. "NEREL: A Russian Dataset with Nested Named Entities and Relations." In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021) (pp. 876-885). https://aclanthology.org/2021.ranlp-main.100.pdf
