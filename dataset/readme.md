# Datsets

## [KONECT](http://konect.cc/)

```bibtex
@misc{konect, 
    url={http://konect.cc/}, 
    journal={Konect.cc}, 
    year={2023}
}
```

### Zachary karate club

[kar](http://konect.cc/networks/ucidata-zachary/): This is the well-known and much-used Zachary karate club network. The data was collected from the members of a university karate club by Wayne Zachary in 1977. Each node represents a member of the club, and each edge represents a tie between two members of the club. The network is undirected. An often discussed problem using this dataset is to find the two groups of people into which the karate club split after an argument between two teachers.
- Nodes: 34
- Edges: 78

### Dolphins

[dol](http://konect.cc/networks/dolphins/): This is a directed social network of bottlenose dolphins. The nodes are the bottlenose dolphins (genus Tursiops) of a bottlenose dolphin community living off Doubtful Sound, a fjord in New Zealand (spelled fiord in New Zealand). An edge indicates a frequent association. The dolphins were observed between 1994 and 2001.
- Nodes: 62
- Edges: 159

### Train bombing

[mad](http://konect.cc/networks/moreno_train/): This undirected network contains contacts between suspected terrorists involved in the train bombing of Madrid on March 11, 2004 as reconstructed from newspapers. A node represents a terrorist and an edge between two terrorists shows that there was a contact between the two terroists. The edge weights denote how 'strong' a connection was. This includes friendship and co-participating in training camps or previous attacks.
- Nodes: 64
- Edges: 243

### Les Misérables

[lesm](http://konect.cc/networks/moreno_lesmis/): This undirected network contains co-occurances of characters in Victor Hugo's novel 'Les Misérables'. A node represents a character and an edge between two nodes shows that these two characters appeared in the same chapter of the the book. The weight of each link indicates how often such a co-appearance occured.
- Nodes: 77
- Edges: 254

### Political books

[polb](http://konect.cc/networks/dimacs10-polbooks/): This is "[a] network of books about US politics published around the time of the 2004 presidential election and sold by the online bookseller Amazon.com. Edges between books represent frequent copurchasing of books by the same buyers. The network was compiled by V. Krebs and is unpublished, but can found on Krebs' web site (<http://www.orgnet.com/>). Thanks to Valdis Krebs for permission to post these data on this web site."
- Nodes: 105
- Edges: 441

### David Copperfield

[words](http://konect.cc/networks/adjnoun_adjacency/): This is the undirected network of common noun and adjective adjacencies for the novel David Copperfield by English 19th century writer Charles Dickens. A node represents either a noun or an adjective. An edge connects two words that occur in adjacent positions. The network is not bipartite, i.e., there are edges connecting adjectives with adjectives, nouns with nouns and adjectives with nouns.
- Nodes: 112
- Edges: 425

### Erdős

[erdos](http://konect.cc/networks/pajek-erdos/): This is the co-authorship graph around Paul Erdős. The network is as of 2002, and contains people who have, directly and indirectly, written papers with Paul Erdős. This network is used to define the "Erdős number", i.e., the distance between any node and Paul Erdős. This dataset was assembled by the Pajek project; we do not know the extent of data that is included.
- Nodes: 6927
- Edges: 11850

### US power grid

[pow](http://konect.cc/networks/opsahl-powergrid/):  This undirected network contains information about the power grid of the Western States of the United States of America. An edge represents a power supply line. A node is either a generator, a transformator or a substation.
- Nodes: 4941
- Edges: 6594
- 
## [Network Repository](https://networkrepository.com/index.php)

```bibtex
@inproceedings{nr,
     title={The Network Data Repository with Interactive Graph Analytics and Visualization},
     author={Ryan A. Rossi and Nesreen K. Ahmed},
     booktitle={AAAI},
     url={https://networkrepository.com},
     year={2015}
}
```

### socfb-American75

[fb-75](https://networkrepository.com/socfb-American75.php): A social friendship network extracted from Facebook consisting of people (nodes) with edges representing friendship ties.
- Nodes: 6386
- Edges: 217662

### coauthors-dblp

[dblp](https://networkrepository.com/ca-coauthors-dblp.php): A co-authorship network extracted from DBLP consisting of authors (nodes) with edges representing co-authorships.

## [Network data](http://www-personal.umich.edu/~mejn/netdata/)

```bibtex
@misc{nets, 
    url={http://www-personal.umich.edu/~mejn/netdata/}, 
    journal={Umich.edu}, 
    year={2013} 
}
```

[nets](http://www-personal.umich.edu/~mejn/netdata/netscience.zip): coauthorship network of scientists working on network theory and experiment, as compiled by M. Newman in May 2006. A figure depicting the largest component of this network can be found here. These data can be cited as M. E. J. Newman, Phys. Rev. E 74, 036104 (2006).

- Nodes: 1589
- Edges: 2742

## [Stanford Network Analysis Project](http://snap.stanford.edu/index.html)

```bibtex
@misc{snapnets,
  author       = {Jure Leskovec and Andrej Krevl},
  title        = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
  howpublished = {\url{http://snap.stanford.edu/data}},
  month        = jun,
  year         = 2014
}
```

### Astro Physics collaboration network

[astr](http://snap.stanford.edu/data/ca-AstroPh.html): Arxiv ASTRO-PH (Astro Physics) collaboration network is from the e-print arXiv and covers scientific collaborations between authors papers submitted to Astro Physics category. If an author i co-authored a paper with author j, the graph contains a undirected edge from i to j. If the paper is co-authored by k authors this generates a completely connected (sub)graph on k nodes.

The data covers papers in the period from January 1993 to April 2003 (124 months). It begins within a few months of the inception of the arXiv, and thus represents essentially the complete history of its ASTRO-PH section.

### Amazon product co-purchasing network metadata

[amz](http://snap.stanford.edu/data/amazon-meta.html): The data was collected by crawling Amazon website and contains product metadata and review information about 548,552 different products (Books, music CDs, DVDs and VHS video tapes).

For each product the following information is available:
    - Title
    - Salesrank
    - List of similar products (that get co-purchased with the current product)
    - Detailed product categorization
    - Product reviews: time, customer, rating, number of votes, number of people that found the review helpful
The data was collected in summer 2006.

### Youtube social network and ground-truth communities

[you](http://snap.stanford.edu/data/com-Youtube.html): Youtube is a video-sharing web site that includes a social network. In the Youtube social network, users form friendship each other and users can create groups which other users can join. We consider such user-defined groups as ground-truth communities. This data is provided by Alan Mislove et al.

We regard each connected component in a group as a separate ground-truth community. We remove the ground-truth communities which have less than 3 nodes. We also provide the top 5,000 communities with highest quality which are described in our paper. As for the network, we provide the largest connected component.

### Orkut social network and ground-truth communities

[ork](http://snap.stanford.edu/data/com-Orkut.html): Orkut is a free on-line social network where users form friendship each other. Orkut also allows users form a group which other members can then join. We consider such user-defined groups as ground-truth communities. We provide the Orkut friendship social network and ground-truth communities. This data is provided by Alan Mislove et al.

We regard each connected component in a group as a separate ground-truth community. We remove the ground-truth communities which have less than 3 nodes. We also provide the top 5,000 communities with highest quality which are described in our paper. As for the network, we provide the largest connected component.

## [Network Repository](https://networkrepository.com)

[vote](https://networkrepository.com/soc-wiki-Vote.php): The dataset contains all the Wikipedia voting data from the inception of Wikipedia till January 2008. Nodes in the network represent wikipedia users and a directed edge from node i to node j represents that user i voted on user j.

### No Dataset Found

- [4sq]():
