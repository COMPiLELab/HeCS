function cgls( A, b; tol = 1e-3, maxit = 500000 )
      x0 = zeros( size(A, 2), 1 )
      
      r0 = b - A * x0
      p0 = A' * r0
      s0 = p0

      γ = norm(s0)^2
      β = 2.0
      it = 0 
      for i in 1 : maxit
            it = i
            qi = A * p0
            ai = γ / norm(qi)^2
            x0 = x0 + ai * p0
            r0 = r0 - ai * qi

            if norm(r0, Inf) < tol
                  break
            end

            s0 = A' * r0
            β = norm( s0 )^2 / γ
            γ = norm( s0 )^2

            p0 = s0 + β * p0
      end

      return x0, it 
end
