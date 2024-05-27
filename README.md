
# House Rocket

House Rocket é uma plataforma digital que tem como modelo de negócio a compra e venda de imóveis utilizando tecnologia.

Você é um Data Scientist contratado pela empresa para ajudar a encontrar as melhores oportunidades de negócio no mercado de imóveis. O CEO da House Rocket gostaria de maximizar a receita da empresa encontrando boas oportunidades de negócio.

## Estratégia

Sua principal estratégia é comprar boas casas em ótimas localizações com preços baixos e revendê-las posteriormente a preços mais altos. Quanto maior a diferença entre a compra e a venda, maior o lucro da empresa e, portanto, maior sua receita.

Entretanto, as casas possuem muitos atributos que as tornam mais ou menos atrativas aos compradores e vendedores, e a localização e o período do ano também podem influenciar os preços.

## Perguntas a serem respondidas

### 1. Qual a correlação entre preço e número de quartos, banheiros, waterfront, sqft_lot, yr_built e afins?

**Resposta:** As variáveis que têm a correlação mais forte com o preço listado são:

- Living Area (sqft)
- Grade
- Above Grade (sqft)
- Living Area 15 (sqft)
- Número de Banheiros
- Condition

Isso sugere que essas características são fortes indicadoras do preço de uma propriedade. Características como year_built, year_renovated, longitude e lot_area_sqft têm correlações muito baixas com o preço listado, indicando que elas têm um impacto menor no preço da propriedade.

### 2. Qual o tempo médio para realizar uma reforma? Fazer a diferença entre ano construído e ano renovado

**Resposta:** A diferença entre o ano de construção (year_built) e o ano de renovação (year_renovated) é de 56.3 anos.

### 3. Verificar se o preço de venda é menor ou maior quando a reforma é feita

**Resposta:** Se a reforma é feita, o preço da venda é maior, com um aumento de $230,000 no preço de venda.

| Renovated Avg Price | Non-Renovated Avg Price |
|---------------------|-------------------------|
| $760,379.03         | $530,360.82             |

### 4. A relação entre condition e year_renovated é relevante?

**Hipóteses:**

- H0: O year_renovated não depende da condition.
- H1: O year_renovated depende da condition, casas em condição piores devem sofrer renovação.

**Resposta:** O valor p é extremamente pequeno (5.64e-09), muito menor que o nível de significância comum de 0.05. Isso significa que rejeitamos a hipótese nula 𝐻0, indicando que o ano de renovação (year_renovated) depende da condition da casa.

### 5. Quando devemos fazer uma reforma?

**Resposta:**

- **Condição 3 (5.25%):** Esta condição apresenta a maior taxa de renovação, indicando que casas em condição 3 são as mais prováveis de serem renovadas.
- **Condição 1 (3.33%):** Esta é a segunda maior taxa de renovação.
- **Condições 2, 4 e 5 (cerca de 2.2%-2.4%):** Essas condições têm taxas de renovação relativamente menores.

| Condition | Renovation Rate |
|-----------|-----------------|
| 3         | 0.052455        |
| 4         | 0.023948        |
| 1         | 0.033333        |
| 5         | 0.021752        |
| 2         | 0.023256        |

### 6. Quando reformamos 3 e 1, qual seria o preço da venda?

**Resposta:** Casas com condição 3 tem um aumento de $229,086.3 enquanto casas com condição 1 normalmente não são renovadas fica um adendo que há pouca amostra.

| Condition | Renovated Avg Price | Non-Renovated Avg Price | Price Difference | Num Renovated | Num Non-Renovated | Renovation Proportion |
|-----------|---------------------|-------------------------|------------------|---------------|-------------------|-----------------------|
| 1         | $252,000            | $337,274                | -$85,274.1       | 1             | 29                | 0.033333              |
| 3         | $759,082            | $529,996                | $229,086.3       | 736           | 13,295            | 0.052455              |

### Perguntas adicionais

**Quais casas o CEO da House Rocket deveria comprar e por qual preço de compra?**

Para maximizar a receita da House Rocket, recomendamos a compra de casas com características fortemente correlacionadas com o preço, como living_area_sqft, grade, num_bathrooms e condition. Priorize imóveis em condição 3, pois têm maior probabilidade de serem renovados e, portanto, maior potencial de valorização. Negocie preços baixos para aumentar a margem de lucro na revenda. Realize reformas estratégicas em casas nas condições 3 para aumentar significativamente o preço de venda.

A ideia é montar um modelo de ML, aonde a variável preditora é o preço da venda. O modelo deve ser treinado com base nos dados de preço listado, e uma estratégia backward. A estratégia é a seguinte:

1. Treinar o modelo com base nos dados de preço listado.
2. Ajustar o modelo para que ele seja capaz de prever o preço da venda.
3. Considerar as condições para determinar se deve ou não ser renovada, e se for renovada, se o preço da venda é maior ou menor.
4. Ajustar o modelo para que ele seja capaz de prever o preço da venda com base nas condições.

A predição foi realizada utilizando vários modelos, como Random Forest, Gradient Boosting, e Linear Regression.

Com isso feito, as 10 propriedades com maior possibilidade de retorno são:

| property_id  | listing_price | potential_profit_no_renovation | potential_profit_with_renovation |
|--------------|---------------|--------------------------------|----------------------------------|
| 1225069056   | $400,000.00   | $821,617.00                    | $942,912.00                      |
| 9272202240   | $272,000.00   | $548,545.00                    | $744,556.00                      |
| 7899801088   | $209,900.00   | $452,972.00                    | $655,780.00                      |
| 1222029056   | $270,000.00   | $401,541.00                    | $610,946.00                      |
| 8835770368   | $119,900.00   | $572,715.00                    | $590,802.00                      |
| 2625079040   | $299,000.00   | $760,917.00                    | $573,879.00                      |
| 3869900032   | $215,000.00   | $641,226.00                    | $547,304.00                      |
| 3425059072   | $245,000.00   | $510,923.00                    | $528,988.00                      |
| 9406530560   | $110,700.00   | $484,109.00                    | $512,531.00                      |
| 1223089024   | $378,000.00   | $690,325.00                    | $490,146.00                      |

**Uma vez a casa em posse da empresa, qual o melhor momento para vendê-las e qual seria o preço da venda?**

Quando atingir o potencial de lucro.

**A House Rocket deveria fazer uma reforma para aumentar o preço da venda? Quais seriam as sugestões de mudanças? Qual o incremento no preço dado por cada opção de reforma?**

Para decidir se a House Rocket deve realizar reformas em uma propriedade, é importante considerar o estado atual da casa (condição 1 ou 3) e o potencial de lucro após a renovação. As reformas podem agregar um valor significativo ao preço de venda e melhorar o retorno sobre o investimento, mas é importante perceber que existem casas que precisam de reforma mas o potencial de lucro é menor que se considerar a condição atual.
