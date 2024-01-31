using GR: delaunay
using StatsBase

function generateDelauney(N=4)
      points=rand(N, 2)*0.8.+0.1;
      points=[points; [0 0]; [0 1]; [1 0]; [1 1]];
  
      num, tri=delaunay(points[:, 1], points[:, 2]);
      preedges=Array{Integer}(undef, 0, 2);
      trian=Array{Integer}(undef, num, 3);
      
      for i in axes(tri, 1)
          tmp=tri[i, :]; tmp=sort(tmp)'; trian[i, :]=tmp;
          preedges=[preedges; [tmp[1] tmp[2]];     [tmp[1] tmp[3]]; [tmp[2] tmp[3]]];   
      end
  
      preedges=unique(preedges, dims=1);
      preedges=sort(preedges, dims = 2); 
      preedges=sortslices(preedges, dims = 1);
  
      trian=sort(trian, dims = 2); 
      trian=sortslices(trian, dims = 1);  
      return points, preedges, trian
end

getIndx2Kill(edgs::Matrix)=rand(1:size(edgs, 1)-4);

function killEdge(indx, n, edges, trigs)
    out=edges[indx,:];
    edges2=edges[1:size(edges,1) .!= indx, : ];
    out_trigs=Vector{Integer}();
    for i in axes(trigs, 1)
        trig=trigs[i,:];
        if ((trig[1] == out[1]) && (trig[2] == out[2])) || ((trig[1] == out[1]) && (trig[3] == out[2])) || ((trig[2] == out[1]) && (trig[3] == out[2]))
            out_trigs=[out_trigs; i];
        end
    end
    trigs2=trigs[setdiff(1:size(trigs, 1), out_trigs) ,:];
    return edges2, trigs2
end

function getNewEdge2(n, edges2, allEdges)
    new_edge = allEdges[ sample( 1:size(allEdges, 1), 1), :]
    indx = findall(all(allEdges .== new_edge, dims=2))[1][1]
    allEdges = allEdges[ 1:size(allEdges, 1) .!= indx, : ]
    return new_edge, allEdges
end

function addEdge(new_edge, n, edges, trigs)
    edges2=[edges; new_edge];
    edges2=sort(edges2, dims = 2); 
    edges2=sortslices(edges2, dims = 1);

    new_trigs=Array{Integer}(undef, 0, 3);
    for i in 1:n
        if (i != new_edge[1]) && (i != new_edge[2])
            if (sum(all(edges .== sort([new_edge[1]; i])', dims=2))==1) && (sum(all(edges .== sort([new_edge[2]; i])', dims=2))==1)
               new_trigs=[new_trigs; sort([new_edge i], dims=1)];
           end
        end
    end
    trigs2=[trigs; new_trigs];
    trigs2=sort(trigs2, dims = 2); 
    trigs2=sortslices(trigs2, dims = 1);
    return edges2, trigs2
end

function sparseDelaunay( ; N = 10, ν = 0.4 )
      n  = N + 4;

      points, edges, trigs = generateDelauney( N )
      ν_init = size(edges, 1 ) / ( ( N + 4) * ( N + 3 ) / 2 );

      backlash = Int( - size( edges, 1 ) + round( ν * n * (n-1) / 2 ) ) 

      if backlash < 0
            for i in 1 : -backlash
                  ind = getIndx2Kill( edges ) ;
                  edges, trigs = killEdge(ind, n, edges, trigs)
            end
      else
            allEdges = Array{Integer}(undef, 0, 2)
            for i in 1:(n-1) 
                  for j in (i+1):n
                        allEdges = [ allEdges; i j  ]
                  end
            end
            for i in axes(edges, 1)
                  indx = findall(all(allEdges .== edges[i, :]', dims=2))[1][1]
                  allEdges = allEdges[ 1:size(allEdges, 1) .!= indx, : ]
            end
            for i in 1 : backlash      
                  ind, allEdges = getNewEdge2(n, edges, allEdges);
                  edges, trigs = addEdge(ind, n, edges, trigs)
            end
      end

      return n, points, ν_init, edges, trigs

end