# Baselines Algorithms

## Algorithm 2 Random hiding strategy

Hide target node from a target community.
```
Input: Network G, T, Target node n, Target community Ci, Community set 𝐶;
Output: Updated Network 𝐺′;
 
if  n ∈ Ci 
  then
      while 𝑇>0 do
            𝑁𝑛𝐶𝑖← getNonneighborSet(𝐶𝑖,𝑛);
            u← RandomChooseNode(𝑁𝑛𝐶𝑖);
            add link 𝑒(𝑛,𝑣) to E;
            𝐶𝑗← getNeighborSet(𝐶𝑗,𝑛);
            v← RandomChooseNode(𝑁𝑛𝐶𝑗);
            remove link 𝑒(𝑛,𝑣) from E;
            𝑇=𝑇−1
      end
𝐺′← Update G;
end
```

## Algorithm 3 Base-degree hiding strategy

```
Input: Network G, T, Target node n;
Output: Updated Network 𝐺′;

C← getCommunities(G);
[𝑛1,𝑛2,…,𝑛𝑝] ← getNodesInOverlappingArea(C);

if  𝑛∈[𝑛1,𝑛2,…,𝑛𝑝]
  then
      [𝐶1,𝐶2,…,𝐶𝐾]← getCommunitiesOfNode(𝐶,𝑛);
      𝐶𝑖← ChooseTargetCommunity();
      while 𝑇>0 do
            𝑁𝑛𝐶𝑖← getNonneighborSet(𝐶𝑖,𝑛);
            u← ChooseNodeBaseDegree(𝑁𝑛𝐶𝑖);
            add link 𝑒(𝑛,𝑣) to E;
            𝐶𝑗 ← getNeighborSet(𝐶𝑗,𝑛 );
            v← ChooseNodeBaseDegree(𝑁𝑛𝐶𝑗);
            remove link 𝑒(𝑛,𝑣) from E;
            𝑇=𝑇−1
      end
𝐺′← Update G;
end
```
