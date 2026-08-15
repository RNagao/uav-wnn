# Lado Julia do teste diferencial (par de scripts/spec04_diferencial.py).
#
# Roda o `FunctionalDiscriminator` do Katopodis sobre exatamente os mesmos
# bits, alvos, mapeamento de tuplas e ordem de visita que o `NTupleRegressor`
# em Python, e grava as predições para comparação.
#
# O mapeamento é imposto sobrescrevendo `d.segments`/`d.offsets` depois da
# construção: `RawBitsEncoder{UInt16}` dá `res = 16` bits por elemento, então
# o índice global de bit `j` (base 0) vive no elemento `j÷16`, bit `j%16` —
# o mesmo empacotamento little-endian usado do lado Python.
#
# Uso:  julia --project=<env> scripts/diferencial.jl <dir_de_dados>

using Ramnet
using Ramnet.Models.AltDiscriminators: FunctionalDiscriminator, train!, predict
using Ramnet.Encoders: RawBitsEncoder
using Ramnet.Partitioners: indices_to_segment_offset
using DelimitedFiles

const CONFIGS = ["B64_n8_N8", "B96_n12_N8", "B128_n8_N16"]

"Empacota uma linha de bits em palavras de 16, little-endian."
function empacota(linha::AbstractVector{<:Integer}, n_bits::Int)
    palavras = zeros(UInt16, cld(n_bits, 16))
    for j in 0:(n_bits - 1)
        if linha[j + 1] != 0
            palavras[(j >> 4) + 1] |= UInt16(1) << (j & 15)
        end
    end
    return palavras
end

function roda(dir::String)
    meta = parse.(Float64, split(strip(read(joinpath(dir, "meta.txt"), String))))
    n_bits, n_tupla, n_ram, n_saidas = Int.(meta[1:4])
    m_treino, m_teste = Int(meta[5]), Int(meta[6])
    eta = meta[7]

    mapeamento = Int.(vec(readdlm(joinpath(dir, "mapeamento.txt"))))
    bits = Int.(readdlm(joinpath(dir, "bits.txt")))
    alvo = readdlm(joinpath(dir, "alvo.txt"), Float64)
    ordem = Int.(vec(readdlm(joinpath(dir, "ordem.txt"))))

    input_len = cld(n_bits, 16)
    enc = RawBitsEncoder{UInt16}()
    d = FunctionalDiscriminator{n_saidas}(input_len, n_tupla, enc, :uniform_random)

    length(d.nodes) == n_ram || error(
        "esperava $n_ram nós, o partitioner deu $(length(d.nodes)) — B = N·n?")

    # Impõe o mapeamento do lado Python (índices base 0 -> base 1).
    seg, off = indices_to_segment_offset(mapeamento .+ 1, input_len, 16)
    d.segments .= seg
    d.offsets .= off

    X = [empacota(bits[m, :], n_bits) for m in 1:size(bits, 1)]

    for m in ordem                       # `ordem` vem base 0, como o Python usa
        x = X[m + 1]
        f = predict(d, x)
        grad = eta .* (f .- alvo[m + 1, :])
        train!(d, x, grad)
    end

    pred = Matrix{Float64}(undef, m_teste, n_saidas)
    for t in 1:m_teste
        pred[t, :] = predict(d, X[m_treino + t])
    end
    writedlm(joinpath(dir, "pred_julia.txt"), pred)
    println("ok $(basename(dir))  ($(length(d.nodes)) nós, $n_tupla bits/tupla)")
end

base = ARGS[1]
for c in CONFIGS
    roda(joinpath(base, c))
end
