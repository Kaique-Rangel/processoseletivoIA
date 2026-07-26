# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---
## 📝 Relatório do Candidato

👤 **Nome Completo:** Kaique Rangel da Silva

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN foi construída com a Functional API do Keras e é composta por 4 blocos convolucionais, cada um contendo 2 camadas Conv2D (com BatchNormalization e ativação ReLU após cada convolução), seguidas de MaxPooling2D e Dropout. O número de filtros aumenta progressivamente a cada bloco (32 → 64 → 128 → 256), e a taxa de dropout também cresce (0.2 → 0.3 → 0.4 → 0.4) para reforçar a regularização nas camadas mais profundas.

Após os blocos convolucionais, é aplicado GlobalAveragePooling2D (em vez de Flatten, para reduzir o número de parâmetros e o risco de overfitting), seguido de uma camada densa de 128 neurônios com BatchNormalization, Dropout(0.5) e, por fim, a camada de saída com 10 neurônios e ativação softmax.

Todas as camadas convolucionais e densas usam kernel_initializer=RandomNormal(stddev=0.05) e regularização L2 (1e-4). Esse inicializador foi escolhido no lugar de he_normal/glorot_uniform porque, durante os testes no pipeline de CI (GitHub Actions), o model.h5 salvo por uma versão mais nova do Keras não conseguia ser carregado por uma versão mais antiga presente no ambiente de validação — o erro ocorria na deserialização da classe VarianceScaling (base de HeNormal e GlorotUniform), que passou a serializar parâmetros (input_axes, output_axes) não reconhecidos por versões anteriores. Trocar para RandomNormal, que não pertence a essa família de inicializadores, eliminou o problema de compatibilidade sem impacto relevante na convergência do treino, já que todas as camadas possuem BatchNormalization logo em seguida.

A estratégia de data augmentation foi incorporada diretamente ao modelo (como uma camada keras.Sequential aplicada logo após a entrada), usando: RandomFlip("horizontal"), RandomRotation(0.08), RandomZoom(0.15), RandomTranslation(0.1, 0.1) e RandomContrast(0.1). Por estar embutida no modelo, essa etapa é aplicada automaticamente apenas durante o treino (training=True), sendo ignorada na inferência — o que também simplifica a conversão para TensorFlow Lite.

### 2️⃣ Bibliotecas Utilizadas

- TensorFlow / Keras (tensorflow==2.21.0)
- NumPy

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada Dynamic Range Quantization, aplicada via converter.optimizations = [tf.lite.Optimize.DEFAULT] no TFLiteConverter. Essa técnica quantiza os pesos do modelo (de float32 para uma representação de menor precisão), reduzindo significativamente o tamanho do arquivo final e acelerando a inferência em dispositivos com recursos limitados, sem exigir um dataset representativo para calibração.

Após a conversão, o modelo .tflite foi validado automaticamente dentro do próprio optimize_model.py, carregando-o com tf.lite.Interpreter e conferindo os shapes de entrada e saída, garantindo que o artefato de edge estava íntegro antes de ser usado na etapa de inferência.

### 4️⃣ Resultados Obtidos

- Acurácia de validação: 82,74%
- Acurácia de teste: 81,66%
- Tamanho do model.h5: 14.306,73 KB (~14 MB)
- Tamanho do model.tflite: 1.205,10 KB (~1,2 MB)
- Redução de tamanho após quantização: 91,58%

### 5️⃣ Comentários Adicionais

O treinamento foi feito inteiramente em CPU, o que resultou em um tempo total de aproximadamente 55 minutos para as ~29 épocas executadas até o EarlyStopping interromper o treino (monitorando val_loss, com patience=8). O ReduceLROnPlateau foi essencial para destravar melhorias na segunda metade do treino: a taxa de aprendizado foi reduzida de 1e-3 para 6.25e-5 ao longo de 4 reduções, e as maiores melhorias de acurácia de validação ocorreram logo após essas reduções.

A principal decisão técnica foi usar 2 convoluções por bloco (em vez de apenas 1), o que aumenta a capacidade de extração de features de cada bloco antes do downsampling, mantendo ainda assim uma arquitetura compacta (8 camadas convolucionais no total) — dentro do espírito de "CNN simples" pedido no desafio, sem chegar perto da profundidade de arquiteturas como ResNet ou VGG.

O run_inference.py foi fornecido como parte do template do projeto e foi utilizado sem modificações, após validado contra o model.tflite gerado.

Durante a etapa de validação automática (CI/GitHub Actions), o carregamento do model.h5 falhou com o erro `HeNormal.__init__() got an unexpected keyword argument 'input_axes'`. A causa foi um descompasso de versões do Keras entre o ambiente local (usado para treinar e salvar o modelo) e o ambiente de validação: versões mais novas do Keras serializam o inicializador HeNormal (e, por herdarem da mesma classe base VarianceScaling, também GlorotUniform) com parâmetros adicionais que versões mais antigas não reconhecem ao desserializar. A correção foi trocar o kernel_initializer de todas as camadas para RandomNormal(stddev=0.05), que não pertence a essa família e serializa de forma estável entre versões, eliminando o problema sem necessidade de alterar a arquitetura do modelo. Também foi fixada a versão exata do TensorFlow no requirements.txt (tensorflow==2.21.0) para reduzir o risco de outras incompatibilidades de ambiente.

### 6️⃣ Exemplo de Inferência

```
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=ship | real=airplane
Amostra 5: predito=frog | real=frog
```

4 das 5 amostras testadas foram classificadas corretamente (80%, em linha com a acurácia de teste obtida). O único erro foi a amostra 4, onde o modelo previu "ship" para uma imagem cuja classe real é "airplane" — uma confusão que faz sentido visualmente, já que aviões e navios em baixa resolução (32x32) costumam compartilhar fundo predominantemente azul/céu ou água, além de silhuetas alongadas, o que pode levar a CNN a confundir as duas classes nesse tipo de amostra.