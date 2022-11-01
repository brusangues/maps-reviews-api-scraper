# google maps review api scraper
Um scraper para reviews do google maps, focado em hotéis.
Este scraper utiliza a api interna reviewDialog, e não necessita de drivers
interativos como o Selenium.

# Comandos
Para rodar sequencialmente:  
```python -m app run --path input/test.csv```

Para rodar com múltiplas threads em paralelo:  
```python -m app run-async --path input/hotels.csv```

# Manual de utilização do scraper localmente no windows

Para a instalação do python, recomendo a utilização do miniconda, para utilização de ambientes virtuais, como mostro no passo a passo abaixo.

1. Instalação do miniconda
    1. https://docs.conda.io/en/latest/miniconda.html
    2. Utilizar primeiro instalador para windows, ou um instalador com versão python 3.9
    3. Durante a instalação, adicionar o miniconda ao path
2. Teste da instalação do miniconda
    1. Se tiver terminais abertos, feche-os
    2. Abra um terminal (cmd)
    3. Teste o comando ```conda -V```, que deve retornar algo do tipo: ```conda 4.10.3```
3. Criação do ambiente conda
    1. Em um terminal, mude para a pasta raiz deste repositório
    2. Crie o ambiente scraping com o comando ```conda create -n scraping python=3.9```
    3. Entre no ambiente com o comando ```conda activate scraping```
    4. Teste a instalação do python com o comando ```python -V```, que deve retornar algo do tipo: ```Python 3.9.13```
4. Instalação dos pacotes do projeto
    1. Se já não estiver ativado, ative o ambiente com o comando ```conda activate scraping```
    2. Instale os pacotes usando o comando ```pip install -r requirements.txt```
5. Teste toda a aplicação
    1. Se já não estiver ativado, ative o ambiente com o comando ```conda activate scraping```
    2. Utilize o comando para testar a aplicação ```python -m app run --path input/test.csv```
    3. Verifique os reviews raspados em ```data/ano/mes/dia/*.csv```
    4. Verifique os metadados do hotel em ```data/places.csv```
