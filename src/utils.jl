
"""
    indicator( ind, m )

      returns the indicator (subsampling) diagonal matrix with 1's on the main diagonal in the places from `ind`
"""
function indicator( ind, m )
      res = zeros(m)
      res[ind] .= 1
      return diagm(res)
end

"""
    Perm(v)

      Returns the permutation matrix corresponding to the permutation `v`
"""
function Perm(v)
      m = size(v, 1)
      res = zeros( m, m )
      for i in 1 : m 
            res[ v[i], i ] = 1
      end
      return res
end

"""
    condPlus( A; thr = 1e-4 )

      Return the condition number of the matrix _in the least square sense_.
"""
function condPlus( A; thr = 1e-4 )
      m = size(A, 1);
      σ = svd(Matrix(A)).S;
      return maximum( σ ) / minimum( σ[ abs.(σ) .> thr ])
end


"""
    getAllEdges( n )

      generate all possible edges in the graph of `n` vertices
"""
function getAllEdges( n )
      allEdges = Array{Integer}(undef, 0, 2)
      for i in 1:(n-1) 
            for j in (i+1):n
                  allEdges = [ allEdges; i j  ]
            end
      end
      return allEdges
end


"""
    myEig2(L1up)

      Computationally more efficient and robust condition number
"""
function myEig2(L1up)
      Afun(x) = Krylov.lsmr(L1up, x, ldiv = true, rtol=1e-16, itmax=10^8)[1]
      m = size(L1up, 1);
      D = LinearMap(
            Afun, Afun, m, m; ismutating = false, issymmetric = true
      );
      decomp,  = partialschur(D, nev=1, tol=1e-16, which=LM());
      λs_inv, X = partialeigen(decomp);
      Afun2(x) = L1up * x;
      D2 = LinearMap(
            Afun2, Afun2, m, m; ismutating = false, issymmetric = true
      );
      decomp2,  = partialschur(D2, nev=1, tol=1e-6, which=LM());
      λs, X= partialeigen(decomp2);
      return λs_inv[1]*λs[1]
end
