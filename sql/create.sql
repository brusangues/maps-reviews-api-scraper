CREATE TABLE avaliacoes_hoteis (
    id bigserial PRIMARY KEY,
    hotel varchar(1000),
    estrelas int,
    regiao varchar(100),
    estado varchar(10),
    nota float,
    texto text,
    embedding vector(768)
);
