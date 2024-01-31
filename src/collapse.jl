function greedyCollapse( edges, trigs )
      edge2Trig = getEdges2Trig(edges, trigs)
      trig2Edg = getTrig2Edg(edges, trigs, edge2Trig)

      Ls = [ length(edge2Trig[i]) for i in 1 : size( edges, 1 ) ]
      Free = Set([ i for i in 1 : size(edges, 1 ) if Ls[i] == 1 ])

      Σ = [];
      Τ = [ ]; 
      while !isempty(Free)
            σ = pop!(Free) 
            Σ = [ Σ; σ ]
            τ = first( edge2Trig[ σ ] ) 
            Τ = [ Τ ; τ ]
            for ξ in trig2Edg[ τ ]
                  setdiff!( edge2Trig[ ξ ], Set([ τ ]) )
                  Ls[ ξ ] = Ls[ ξ ] - 1
                  ( Ls[ ξ ] == 1 ) && ( union!(Free, Set( [ ξ ] ) ) )
                  ( Ls[ ξ ] == 0 ) && ( setdiff!(Free, Set( [ ξ ] ) ) )
            end
      end
      return sum( Ls ) == 0, Σ, Τ, edge2Trig, trig2Edg, Ls, Free  
end

function greedyCollapseShort( edge2Trig, trig2Edg, perm, edges)
      
      Ls = [ length(edge2Trig[i]) for i in 1 : size( edges, 1 ) ]
      Free = Set([ i for i in 1 : size(edges, 1 ) if Ls[i] == 1 ])

      Σ = [];
      Τ = [ ]; 
      while !isempty(Free)
            σ = pop!(Free) 
            Σ = [ Σ; σ ]
            τ = first( edge2Trig[ σ ] ) 
            Τ = [ Τ ; τ ]
            in = findall(perm .== τ)[1][1] 
            for ξ in trig2Edg[ in ]
                  setdiff!( edge2Trig[ ξ ], Set([ τ ]) )
                  Ls[ ξ ] = Ls[ ξ ] - 1
                  ( Ls[ ξ ] == 1 ) && ( union!(Free, Set( [ ξ ] ) ) )
                  ( Ls[ ξ ] == 0 ) && ( setdiff!(Free, Set( [ ξ ] ) ) )
            end
      end
      return sum( Ls ) == 0, Σ, Τ, edge2Trig, trig2Edg, Ls, Free  
end