# Lado Julia do teste diferencial das redes da dissertação
# (par de scripts/spec04_diferencial_redes.py).
#
# Roda `FunctionalDiscriminator` (§4.3.1) e `RegressionDiscriminator` (§4.2.1,
# com fator de esquecimento) sobre exatamente o mesmo particionamento de bits,
# os mesmos dados e a mesma sequência de atualizações do lado Python.
#
# Uso:  julia --project=<env> scripts/diferencial_redes.jl <dir_de_dados>

using Ramnet
using Ramnet.Models.AltDiscriminators: FunctionalDiscriminator,
                                       RegressionDiscriminator, train!, predict
using Ramnet.Encoders: RawBitsEncoder
using Ramnet.Partitioners: indices_to_segment_offset
using StaticArrays
using DelimitedFiles

const CONFIGS = ["ator_B64_n8", "ator_B96_n12",
                 "critico_B64_n8_phi085", "critico_B96_n12_phi100",
                 "critico_B64_n8_phi050"]

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
    campos = split(strip(read(joinpath(dir, "meta.txt"), String)))
    n_bits, n_tupla, n_ram, n_saidas = parse.(Int, campos[1:4])
    m_treino, m_teste = parse.(Int, campos[5:6])
    eta = parse(Float64, campos[7])
    papel = String(campos[8])
    phi = parse(Float64, campos[9])

    mapeamento = Int.(vec(readdlm(joinpath(dir, "mapeamento.txt"))))
    bits = Int.(readdlm(joinpath(dir, "bits.txt")))
    alvo = readdlm(joinpath(dir, "alvo.txt"), Float64)

    input_len = cld(n_bits, 16)
    enc = RawBitsEncoder{UInt16}()

    d = papel == "ator" ?
        FunctionalDiscriminator{n_saidas}(input_len, n_tupla, enc, :uniform_random) :
        RegressionDiscriminator{n_saidas}(input_len, n_tupla, enc, :uniform_random; γ=phi)

    length(d.nodes) == n_ram || error(
        "esperava $n_ram nós, o partitioner deu $(length(d.nodes))")

    # Impõe a MESMA partição do lado Python (índices base 0 -> base 1).
    seg, off = indices_to_segment_offset(mapeamento .+ 1, input_len, 16)
    d.segments .= seg
    d.offsets .= off

    X = [empacota(bits[m, :], n_bits) for m in 1:size(bits, 1)]

    for m in 1:m_treino
        x = X[m]
        y = SVector{n_saidas,Float64}(alvo[m, :])
        if papel == "ator"
            # eq. (4.11) com passo -eta*(f - y); train! soma -grad, logo grad = eta*(f-y)
            f = predict(d, x)
            train!(d, x, SVector{n_saidas,Float64}(eta .* (f .- alvo[m, :])))
        else
            train!(d, x, y)                       # eq. (4.8)
        end
    end

    pred = Matrix{Float64}(undef, m_teste, n_saidas)
    for t in 1:m_teste
        pred[t, :] = predict(d, X[m_treino + t])
    end
    writedlm(joinpath(dir, "pred_julia.txt"), pred)
    println("ok $(basename(dir))  ($papel, $(length(d.nodes)) nós, φ=$phi)")
end

base = ARGS[1]
for c in CONFIGS
    roda(joinpath(base, c))
end
