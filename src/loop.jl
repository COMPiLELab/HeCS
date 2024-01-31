using GR: delaunay  # Delaunay triangulation
using StatsBase, Distributions # sampling
using LinearAlgebra, Arpack, Random, SparseArrays # LA
using ArnoldiMethod, Krylov, LinearMaps # eig in LA
using BenchmarkTools, Printf, TimerOutputs # timings and staff

# and plotting
using Plots, ColorSchemes, Plots.PlotMeasures,  LaTeXStrings
pgfplotsx()
theme(:mute)
Plots.scalefontsizes(1.75)
cols=ColorSchemes.Spectral_11;

include("utils.jl")
include("cgls.jl")
include("generateDelaunay.jl")
include("simplicialComplex.jl")
include("collapse.jl")

function repeatTries(N, add, rep)
      κ_original = zeros( rep )
      κ_precon = zeros( rep )
      κ_ldl = zeros( rep )
      it_ldl = zeros( rep )
      it_original = zeros( rep )
      it_precon = zeros( rep )
      m_sizes = zeros( rep )
      Δ_sizes = zeros( rep )

      for repIt in 1 : rep
            points, edges, trigs = generateDelauney( N )
            edges2, trigs2 = deepcopy(edges), deepcopy(trigs)
            n = N + 4
            
            allEdges = getAllEdges(n)
            for i in axes(edges2, 1)
                  indx = findall(all(allEdges .== edges2[i, :]', dims=2))[1][1]
                  allEdges = allEdges[ 1:size(allEdges, 1) .!= indx, : ]
            end
            for i in 1 : add      
                  ind, allEdges = getNewEdge2(n, edges2, allEdges);
                  edges2, trigs2 = addEdge(ind, n, edges2, trigs2)
            end


            w = zeros( size(trigs2, 1), 1 )
            m = size( edges2, 1 )
            Δ = size( trigs2, 1 )

            m_sizes[ repIt ] = m
            Δ_sizes[ repIt ] = Δ

            if size(Δ, 1) > 0.95*m/4*log(4*m)
                  break
            end

            edg2Trig = getEdges2Trig( edges2, trigs2 )
            trig2Edge = getTrig2Edg( edges2, trigs2, edg2Trig )

            w_e = abs.( randn( size(edges2, 1) ) )
            for i in 1 : Δ
                  w[i] = minimum( w_e[ collect( trig2Edge[i] ) ]  )
            end

            W = diagm(vec(sqrt.(w)))
            B2 = B2fromTrig( edges2, trigs2 )
            Lu =  B2 * W * W * B2'



            perm = sortperm( w, dims =1, rev = true )      
            sub = [ ] 
            ind = 1

            subEdg2Trigs = [ Set([ ]) for i in 1 : m ]
            subTrig2Edg = [ ]

            while ind <= Δ
                  tmpSubEdg2Trigs = deepcopy( subEdg2Trigs )
                  tmpSubTrig2Edg = deepcopy( subTrig2Edg )
                  tmpSub = deepcopy( sub )

                  tmpSub = [ tmpSub; perm[ind] ]
                  tmpSubTrig2Edg = [ tmpSubTrig2Edg; trig2Edge[ perm[ind] ] ]
                  for e in trig2Edge[ perm[ind] ]
                        tmpSubEdg2Trigs[e] = union( tmpSubEdg2Trigs[e], perm[ind] )
                  end

                  fl, Σ, Τ, edge2Trig, trig2Edg, Ls, Free = greedyCollapseShort( tmpSubEdg2Trigs, tmpSubTrig2Edg, tmpSub, edges2 )
                  
                  if fl
                        sub = [ sub; perm[ind] ]
                        subTrig2Edg = [ subTrig2Edg; trig2Edge[ perm[ind] ] ]
                        for e in trig2Edge[ perm[ind] ]
                              subEdg2Trigs[e] = union( subEdg2Trigs[e], perm[ind] )
                        end
                  end

                  ind = ind + 1
            end

            filter = sub
            Π = indicator( sort(filter), Δ )
            trigs3 = trigs2[ sort(filter), :]
            edge2Trig2 = getEdges2Trig( edges2, trigs3 )
            trig2Edge2 = getTrig2Edg( edges2, trigs3, edge2Trig2 )

            fl, Σ, Τ, edge2Trig2, trig2Edg2, Ls, Free = greedyCollapse( edges2, trigs3 )

            Σ_full = [ Σ; sort(collect(setdiff( Set(1:m), Set(Σ)))) ]
            Τ_full = [ filter[Τ]; sort(collect(setdiff( Set(1:Δ), Set(filter[Τ]))))  ]

            P1 = Perm( Σ_full )
            P2 = Perm( Τ_full )

            C = P1 * B2 * W * Π * P2
            
            Lu2 = pinv(C) * P1 * Lu * P1' * pinv(C')
            Lu2 = Lu2[1:size(sub,1), 1:size(sub,1)]
            κ_original[ repIt ] = myEig2(sparse(Lu))
            κ_precon[ repIt ] = myEig2(sparse(Lu2))
            _, it_original[ repIt ] = cgls(Lu, Lu*ones( size(Lu, 1) ))
            _, it_precon[ repIt ] = cgls(Lu2, Lu2*ones( size(Lu2, 1) ))
      end

      return κ_original, κ_precon, κ_ldl, it_original, it_precon, it_ldl, m_sizes, Δ_sizes
end


function addCycle(N, rep; maxAdd = 50, δ=2)
      κOs = Array{Float64}(undef, rep, 0)
      κPs = Array{Float64}(undef, rep, 0)
      κLs = Array{Float64}(undef, rep, 0)
      itOs = Array{Float64}(undef, rep, 0)
      itPs = Array{Float64}(undef, rep, 0)
      itLs = Array{Float64}(undef, rep, 0)
      ms = Array{Float64}(undef, rep, 0)
      Δs = Array{Float64}(undef, rep, 0)
      add = 0
      while add <= maxAdd
            add = add + δ
            κ_original, κ_precon, κ_ldl, it_original, it_precon, it_ldl, m_sizes, Δ_sizes = repeatTries(N, add, rep);
            if (add > maxAdd)
                  break
            end 
            κOs = [ κOs κ_original ] 
            κPs = [ κPs κ_precon ]
            κLs = [ κLs κ_ldl ]  
            itOs = [ itOs it_original ] 
            itPs = [ itPs it_precon ] 
            itLs = [ itLs it_ldl ] 
            ms = [ms m_sizes]
            Δs = [Δs Δ_sizes ]
            @printf "added: %i \n" add
      end
      
      return κOs, κPs, κLs, itOs, itPs, itLs, ms, Δs
end

rep = 10

κOs25, κPs25, κLs25, itOs25, itPs25, itLs25, ms25, Δs25 = addCycle(21, rep; maxAdd = 60, δ = 6);

κOs50, κPs50, κLs50, itOs50, itPs50, itLs50, ms50, Δs50 = addCycle(46, rep; maxAdd = 250, δ = 25);

κOs100, κPs100, κLs100, itOs100, itPs100, itLs100, ms100, Δs100 = addCycle(96, rep; maxAdd = 900, δ = 90);

κOs200, κPs200, κLs200, itOs200, itPs200, itLs200, ms200, Δs200 = addCycle(196, rep; maxAdd = 3000, δ = 300);
