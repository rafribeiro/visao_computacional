# Primeira tentativa

Abordagem simples em que inicialmente o fundo é capturado e em seguida cada frame capturado é comparado com o fundo.

Pixels em que o valor de cada componente RGB seja diferente do fundo por mais do que um limiar são mantidos e pixels abaixo do limiar são substituídos pela imagem do fundo virtual.