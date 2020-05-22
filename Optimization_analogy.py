from Transformer import Pairwise_transformer

def build_graph(GDC,AGDC,PGDC,Analogy_GDC,Analogy_AGDC,Analogy_PGDC):

    n1 = len(PGDC)
    n2 = len(Analogy_PGDC)

    n = n1
    if n2 > n1:
        n = n2
    
    matrix_concepts = [[0 for i in range(n1)] for i in range(n)]
    matrix_analogies = [[0 for i in range(n1)] for i in range(n)]

    for i in range(n1):
        matrix_concepts[i][i] = 1
        if i != n1-1:
            matrix_concepts[i][i+1] = 1
            continue

    for i in range(n2):
        matrix_analogies[i][i] = 1
        if i != n2-1:
            matrix_analogies[i][i+1] = 1
            continue

    return (matrix_concepts,matrix_analogies)
    

def get_concepts_and_analogies():

    #========== CONVEX OPTIMIZATION ANALOGIES =================

    #1. GRADIENT DESCENT

    GDC = ["A convex function has a single stationary point.",
           "One minima or maxima.",
           "The gradient of that function gives us the direction of maximum increase of the function, in its local neighborhood",
           "Since we want to find the minima, we take a step in the negative gradient direction."
           "We then reach a point that has lower value than where we where previously, the new point being in the local neighborhood of the previous point.",
           "The actual minima maybe far away from the point where we started.",
           "But, the gradient information about increase is only true in a local neighborhood.",
           "So, we repeat the process of moving a small step in the negative gradient direction until we reach the minima."]

    Analogy_GDC = ["Imagine we were on the surface of the ocean and it was completely transparent.",
                   "We can see clean through to the ocean floor.",
                   "Now, lets imagine the ocean is at its deepest when looked down from the surface, standing at some point on the surface.",
                   "We want to find the point on the surface, where the ocean is deepest below us.",
                   "Our function now is a elevation map.",
                   "We know that if we moved along a certain direction on the surface, the elevation increases locally.",
                   "This is the gradient.",
                   "We want to know where the elevation decreases, not increases.",
                   "This is because we are looking for the deepest part of the ocean.",
                   "So, we move in the direction of the negative gradient that corresponds to an elevation decrease locally.",
                   "Since this direction can only be trusted locally, we move very little everytime we take a step in this direction.",
                   "We repeat this process until we reach the point on the surface, where the ocean is deepest below us."]

    #2. ACCELERATED GRADIENT DESCENT

    AGDC = ["The reason the gradient can be trusted only locally, is because the gradient gives information about the local linear relationship.",
            "Functions are not linear over the entire volume of their domain.",
            "But, differentiable convex functions are locally linear.",
            "Sometimes, it can happen that you may have information that tells you that the function is showing linearity in a broader neighborhood than the local neighborhood of a point",
            "This information is specific to a particular point.",
            "We would like to be able to use this information, when available to us at any given point.",
            "We would like to move more, instead of very little in the negative gradient direction, when we have this type of information.",
            "Availability of this information, in mathematics comes in the form of evaluating the hessian.",
            "This encapsulates second derivative information in a matrix.",
            "Roughly, this says how fast is the gradient changing in the local neighborhood.",
            "Since, the gradient is a measure of local linear relationship, if the hessian is small, the linearity is not chaning much.",
            "Hence, we can take the risk of moving faster at these points.",
            "This is the idea of accelerated gradient descent."]

    Analogy_AGDC = ["In addition to our elevation map and the elevation gradient, we now have a new device.",
                    "This device measures how the elevation gradient locally changes beyond the local neighborhood but, not too far beyond.",
                    "If the signal on the device is low, It means the elevation gradient is not supposed to change beyond the local neighborhood.",
                    "This information is specific to point on the ocean surface that we are on.",
                    "So, when we step away from the surface in the direction of the negative elevation gradient, we can move faster.",
                    "This is because, equipped with this new device, we know that the elevation map is showing linear behavior a little bit beyond the local neighborhood."]

    #3. PROJECTED GRADIENT DESCENT

    PGDC = ["We now are constrained to stay only in a particular convex region in the domain of our function.",
            "We are interested in finding the minima, within this restricted domain.",
            "We follow the same steps as before, the difference is that when we find the next point is beyond the restricted domain, we find a projection to the closest allowable point.",
            "So, everytime we calculate the negative gradient direction, if the new point is within the restricted domain, we take a step in that direction.",
            "If, the new point is outside the restricted domain, we compute a projection to the closest point in the restricted domain and take a step towards the new point."]

    Analogy_PGDC = ["Now, certain points on the ocean surface are forbidden, cause its a different country beyond that point.",
                    "We are given a device that shows us if the direction of negative elevation gradient can lead to border transgression.",
                    "If the device beeps, it gives us a safe point that is closest to the point that would cause the violation.",
                    "We now proceed to move a little bit towards that safe point without worrying about border violations."]


    AGDC = GDC + AGDC
    PGDC = AGDC + PGDC
    Analogy_AGDC = Analogy_GDC + Analogy_AGDC
    Analogy_PGDC = Analogy_AGDC + Analogy_PGDC

    #DEBUGGING: print (len(Analogy_PGDC))

    #DEBUGGING: print (len(PGDC))

    #===================== ENCODING THE DATA FOR THE TRANSFORMER =====================================

    graphs = build_graph(GDC,AGDC,PGDC,Analogy_GDC,Analogy_AGDC,Analogy_PGDC)

    X1 = [[] for i in range(3)]
    X2 = [[] for i in range(3)]

    concept_graph,analogy_graph = graphs[0],graphs[1]


    for sentence in GDC:
        pos = PGDC.index(sentence)
        sentence_encoding = concept_graph[pos]
        X1[0].append(sentence_encoding)
    pad = [0 for i in range(len(X1[0][0]))]
    n_pads = len(PGDC) - len(GDC)
    for j in range(n_pads):
        X1[0].append(pad)

    for sentence in Analogy_GDC:
        pos = Analogy_PGDC.index(sentence)
        sentence_encoding = analogy_graph[pos]
        X2[0].append(sentence_encoding)
    n_pads = len(PGDC) - len(Analogy_GDC)
    for j in range(n_pads):
        X2[0].append(pad)

    for sentence in AGDC:
        pos = PGDC.index(sentence)
        sentence_encoding = concept_graph[pos]
        X1[1].append(sentence_encoding)
    n_pads = len(PGDC) - len(AGDC)
    for j in range(n_pads):
        X1[1].append(pad)

    for sentence in Analogy_AGDC:
        pos = Analogy_PGDC.index(sentence)
        sentence_encoding = analogy_graph[pos]
        X2[1].append(sentence_encoding)
    n_pads = len(PGDC) - len(Analogy_AGDC)
    for j in range(n_pads):
        X2[1].append(pad)

    for sentence in PGDC:
        pos = PGDC.index(sentence)
        sentence_encoding = concept_graph[pos]
        X1[2].append(sentence_encoding)
    n_pads = len(PGDC) - len(PGDC)
    for j in range(n_pads):
        X1[2].append(pad)

    for sentence in Analogy_PGDC:
        pos = Analogy_PGDC.index(sentence)
        sentence_encoding = analogy_graph[pos]
        X2[2].append(sentence_encoding)
    n_pads = len(PGDC) - len(Analogy_PGDC)
    for j in range(n_pads):
        X2[2].append(pad)


    return (X1,X2)

def main():
    """main method
       starting point of prg
    """

    data = get_concepts_and_analogies()
    clf = Pairwise_transformer(blocks = (1,1))
    clf.train(data[0],data[1])

main()
    

    
