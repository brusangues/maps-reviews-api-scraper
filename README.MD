# google maps review api scraper
Um scraper para reviews do google maps, focado em hotéis.
Este scraper utiliza a api interna reviewDialog, e não necessita de drivers
interativos como o Selenium.

# Comandos
Para rodar sequencialmente:  
`python -m app run --path input/test.csv`

Para rodar com múltiplas threads em paralelo:  
`python -m app run-async --path input/hotels.csv`

# Input
O csv de input deve ser preenchido com os seguintes campos obrigatórios:
 - done: Flag 0 ou 1 indicando se a linha já foi processada.
Alterar manualmente.
 - name: Nome arbitrário para o hotel/linha.
 - n_reviews: Número de reviews a ser buscado. 
Pode exceder o máximo, pois, neste caso, o scraper irá parar automaticamente. 
Se o número for negativo, o scraper busca todos os reviews possíveis.
 - sort_by: Tipo de ordenação. Opções disponíveis:
`{"most_relevant", "newest", "highest_rating", "lowest_rating"}`
 - hl: Código de idioma. Representa o idioma base dos reviews buscados. 
Atualmente o scraper dá suporte aos seguintes idiomas: `{"pt-br","en"}`
 - url: Url base para a busca. 

**Para obter o url, pesquise no google o nome do**
**hotel + maps, e encontre o link do google maps para o hotel.**
**Copie e cole o primeiro url que encontrar.**
Esse url deverá conter o feature_id do lugar, representado por dois números 
hexadecimais separados por dois pontos, como `0x0:0x6825759cc04b2504`. Exemplo: `https://www.google.com/maps/place/Pousada+Itarar%C3%A9/@-23.9681059,-46.357305,15z/data=!4m2!3m1!1s0x0:0x6825759cc04b2504?sa=...`


Demais campos nos arquivos de input não são obrigatórios, e não impactam na 
execução do scraper, servindo apenas de controle.

# Comandos para utilização do scraper no Google Colab

Para rodar no ambiente em nuvem Google Colab, acesse diretamente o 
**[Jupyter Notebook deste repositório no Colab](https://colab.research.google.com/github/brusangues/maps-reviews-api-scraper/blob/master/maps_reviews_api_scraper.ipynb)**. 

**Lembre-se de alterar o input e baixar os dados depois de rodar o scraper, pois 
o ambiente será reiniciado e os dados serão perdidos caso a aba não fique ativa 
por muito tempo.**

Alternativamente, pode-se criar um novo notebook com os comandos a seguir:

1. Clonando o repositório
`!git clone https://github.com/brusangues/maps-reviews-api-scraper`
2. Entrando na pasta raiz do repositório
`%cd maps-reviews-api-scraper`
3. Instalando pacotes  
`!pip install -r requirements.txt`
4. Rodando o scraper  
`!python -m app run-async --path input/test.csv`


# Manual de utilização do scraper localmente no windows

Para a instalação do python, recomendo a utilização do miniconda, para 
utilização de ambientes virtuais, como mostro no passo a passo abaixo.

1. Instalação do miniconda
    1. https://docs.conda.io/en/latest/miniconda.html
    2. Utilizar primeiro instalador para windows, ou um instalador com versão python 3.9
    3. Durante a instalação, adicionar o miniconda ao path
2. Teste da instalação do miniconda
    1. Se tiver terminais abertos, feche-os
    2. Abra um terminal (cmd)
    3. Teste o comando `conda -V`, que deve retornar algo do tipo: `conda 4.10.3`
3. Criação do ambiente conda
    1. Em um terminal, mude para a pasta raiz deste repositório
    2. Crie o ambiente scraping com o comando `conda create -n scraping python=3.9`
    3. Entre no ambiente com o comando `conda activate scraping`
    4. Teste a instalação do python com o comando `python -V`, que deve retornar algo do tipo: `Python 3.9.13`
4. Instalação dos pacotes do projeto
    1. Se já não estiver ativado, ative o ambiente com o comando `conda activate scraping`
    2. Instale os pacotes usando o comando `pip install -r requirements.txt`
5. Teste toda a aplicação
    1. Se já não estiver ativado, ative o ambiente com o comando `conda activate scraping`
    2. Utilize o comando para testar a aplicação `python -m app run --path input/test.csv`
    3. Verifique os reviews raspados em `data/ano/mes/dia/*.csv`,
com seus respectivos metadados em `data/ano/mes/dia/*.json`
    4. Verifique o consolidado dos metadados em `data/places.csv`
