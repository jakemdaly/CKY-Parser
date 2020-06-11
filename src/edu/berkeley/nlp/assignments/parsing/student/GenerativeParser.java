package edu.berkeley.nlp.assignments.parsing.student;

import edu.berkeley.nlp.assignments.parsing.*;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Indexer;

import java.util.*;

public class GenerativeParser implements Parser{

    /**
     * Generative parser implementing the CKY algorithm, which deploys dynamic programming to ammortize parse time.
     *
     * @author Jake Daly
     *
     */

    public static class GenerativeParserFactory implements ParserFactory {

        public Parser getParser(List<Tree<String>> trainTrees) {
            return new GenerativeParser(trainTrees);
        }
    }

    CounterMap<List<String>, Tree<String>> knownParses;
    CounterMap<Integer, String> spanToCategories;
    SimpleLexicon lexicon;
    Grammar grammar;
    UnaryClosure UC;
    Indexer<String> labelIndexer;
    int number_rules;
    List<String> currentSentence;
    Set<String> allTagsSet;
    Iterator<String> allTagsIter;
    double[][][] scoreU; // Unary (top) chart
    double[][][] scoreB; // Binary (bottom) chart
    BBP[][][] binaryBP; // Binary back pointers
    UBP[][][] unaryBP; // Unary back pointers
    boolean debug = false;

    public GenerativeParser(List<Tree<String>> trainTrees) {
        /**
         * Generative parser which uses CKY algorithm for parsing
         *
         * @param trainTrees
         * @return GenerativeParser
         */

        System.out.print("Annotating / binarizing training trees ... ");
        List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);
        System.out.println("done.");
        System.out.print("Building grammar ... ");

        // Get the grammar (X->Y and X->YZ) and unary closure
        // The unary closure is to prevent infinite recursion when reconstructing the best parse (top -> down)
        grammar = Grammar.generativeGrammarFromTrees(annotatedTrainTrees);
        labelIndexer = grammar.getLabelIndexer();
        number_rules = grammar.getLabelIndexer().size();
        UC = new UnaryClosure(labelIndexer, grammar.getUnaryRules());

        System.out.println("done. (" + grammar.getLabelIndexer().size() + " states)");

        // Lexicon is similar to grammar, but is the mapping of all actual words to parts of speech (aka pre-terminals).
        lexicon = new SimpleLexicon(annotatedTrainTrees);
        allTagsSet = lexicon.getAllTags();
        allTagsIter = allTagsSet.iterator();

        knownParses = new CounterMap<List<String>, Tree<String>>();
        spanToCategories = new CounterMap<Integer, String>();
        for (Tree<String> trainTree : annotatedTrainTrees) {
            List<String> tags = trainTree.getPreTerminalYield();
            knownParses.incrementCount(tags, trainTree, 1.0);
            tallySpans(trainTree, 0);
        }
        System.out.println("done.");
    }


    public Tree<String> getBestParse(List<String> sentence){
        /**
         * Main function implementing the CKY algorithm. Occurs in two parts: (1) bottom-up dynamic program which fills in the probabilities (scores) of both binary and unary charts
         * and (2) top-down recursive reconstruction of the best tree, which follows back pointers starting at ROOT in the unary tree.
         * If there is nothing in the ROOT slot of the top of the unary chart, it will return Tree("ROOT", [])
         *
         * @author Jake Daly
         * @param sentence
         * @return bestTree
         */

        System.out.println("Start getBestParse");
        double then = System.currentTimeMillis();

        currentSentence = sentence;
        int length_sentence = sentence.size();

        Set<Integer> BLeft = new HashSet<Integer>();
        for (BinaryRule r : grammar.getBinaryRules()){
            BLeft.add(r.leftChild);
        }

        // Use binary left children for loop optimization in DP
        ArrayList<Integer> binaryLefts = new ArrayList<>(BLeft);
        BLeft = null;

        // Backpointer arrays used for reconstruction (top-down) of the trees
        unaryBP = new UBP[length_sentence+1][length_sentence+1][number_rules];
        binaryBP = new BBP[length_sentence+1][length_sentence+1][number_rules];
        initializeUnaryBP(unaryBP);
        initializeBinaryBP(binaryBP);

        // Score charts used for ping-ponging back and forth between binary/unary iterations
        scoreU = new double[length_sentence+1][length_sentence+1][number_rules];
        scoreB = new double[length_sentence+1][length_sentence+1][number_rules];
        doubleFill(scoreU, Double.NEGATIVE_INFINITY);
        doubleFill(scoreB, Double.NEGATIVE_INFINITY);


        // Fill bottom row of each chart. Use bottom of binary for pre-terminals because it wouldn't get used otherwise
        for (int i = 0; i < length_sentence; i++) {

            // Binary first
            for (Iterator<String> it = allTagsSet.iterator(); it.hasNext(); ) {
                String label = it.next();
                int parent = labelIndexer.addAndGetIndex(label); // index of the tag within labelIndexer is the ID of the child
                Double score = lexicon.scoreTagging(sentence.get(i), label);
                if(!(Double.isNaN(score)) && !(Double.isInfinite(score))){
                    if (debug) {
                        System.out.println(i + " " + (i+1) + " " + (parent) + "=="+label);
                    }
                    scoreB[i][i+1][parent] = score;
                }
            }

            // Fill in bottom row of unary chart with unary rules (from the closure) that would link these to binary bottom (pre-terminals)
            for (int X = 0; X < number_rules; X++) {
                List<UnaryRule> closedRules = UC.getClosedUnaryRulesByChild(X); // index of the tag within labelIndexer is the ID of the child
                if (!Double.isInfinite(scoreB[i][i+1][X])) {
                    for (UnaryRule UR : closedRules) {
                        int parent = UR.parent;
                        int child = UR.child;
                        if (scoreU[i][i + 1][parent] < UR.getScore() + scoreB[i][i + 1][child]) {
                            if (debug) {
                                double oldScore = scoreU[i][i + 1][parent];
                                System.out.println("Old scoreU["+i+"]["+(i+1)+"]["+parent+"]: "+oldScore
                                        +"... New scoreU["+i+"]["+(i+1)+"]["+parent+"]: "+(UR.getScore() + scoreB[i][i+1][child])
                                + "... Pointer: " + UR.child + "=="+labelIndexer.get(child));
                            }
                            scoreU[i][i + 1][parent] = UR.getScore() + scoreB[i][i + 1][child];
                            unaryBP[i][i+1][parent] = new UBP(UR.parent, UR.child);
                        }
                    }
                }
            }
        }

        /*
        * Start alternating our way upwards, starting with Binary then --> Unary --> Binary --> Unary --> ... --> Unary
        */
        for (int diff = 2; diff <= length_sentence; diff++){
            for (int i = 0; i <= length_sentence-diff; i++){
                int j = i + diff;

                // Binary first: need to try every possible rule to find the best one for each split point
                // Choosing a left child then iterating over all rules, and maxing at index=parent ELIMINATES need to iterate over right rules
                for (int k = i + 1; k < j; k++) { // Loop over all split points
                    for (int r : binaryLefts) { // All potential left children

                        if (scoreU[i][k][r] != Double.NEGATIVE_INFINITY) { // If the left child was never set, we know we can skip this one
                            List<BinaryRule> rulesLeft = grammar.getBinaryRulesByLeftChild(r);
                            for (BinaryRule BR : rulesLeft){
                                if (scoreB[i][j][BR.parent] < BR.getScore() + scoreU[i][k][BR.leftChild] + scoreU[k][j][BR.rightChild]) {
                                    if (debug) {
                                        double oldScore = scoreB[i][j][BR.parent];
                                        System.out.println("Old scoreB["+i+"]["+j+"]["+BR.parent+"]: "+oldScore
                                                +"... New scoreB["+i+"]["+j+"]["+BR.parent+"]: "+(BR.getScore() + scoreU[i][k][BR.leftChild] + scoreU[k][j][BR.rightChild])
                                                + "... left/right/K: " +BR.leftChild+"/"+ BR.rightChild +"/"+k);
                                    }

                                    // Update binary chart
                                    scoreB[i][j][BR.parent] = BR.getScore() + scoreU[i][k][BR.leftChild] + scoreU[k][j][BR.rightChild];
                                    binaryBP[i][j][BR.parent] = new BBP(BR.parent, BR.leftChild, BR.rightChild, k);

                                }
                            }
                        }
                    }
                }

                // Unary rule iteration
                for (int r = 0; r < number_rules; r++) {
                    if (scoreB[i][j][r]!=Double.NEGATIVE_INFINITY) {
                        List<UnaryRule> rules = UC.getClosedUnaryRulesByChild(r);
                        for (UnaryRule UR : rules) {
                            int child = UR.child;
                            int parent = UR.parent;
                            if (scoreU[i][j][parent] < UR.getScore() + scoreB[i][j][child]) {
                                if (debug) {
                                    double oldScore = scoreU[i][j][parent];
                                    System.out.println("Old scoreU["+i+"]["+(j)+"]["+parent+"]: "+oldScore
                                            +"... New scoreU["+i+"]["+(j)+"]["+parent+"]: "+(UR.getScore() + scoreB[i][j][child])
                                            + "... Pointer: " + child + "=="+labelIndexer.get(child));
                                }

                                // Update unary chart
                                scoreU[i][j][parent] = UR.getScore() + scoreB[i][j][child];
                                unaryBP[i][j][parent] = new UBP(parent, child);
                            }
                        }
                    }
                }

            }
        }

        // Chart should now be filled in. The code below will now work it's way from top to bottom to recreate the highest scoring tree with "ROOT" as top
        // If "ROOT" was not 'reached' for this particular sentence, it will return an empty tree with "ROOT" as top
        System.out.println(sentence);
        int rootIdx = labelIndexer.addAndGetIndex("ROOT");
        System.out.println("Score from binary to ROOT "+scoreU[0][length_sentence-1][rootIdx]);
        Tree<String> bestParse;
        if (Double.isInfinite(scoreU[0][length_sentence][rootIdx])){
            bestParse = new Tree<>("ROOT");
        } else {
            List<Tree<String>> treeList = Arrays.asList(getBestTreeUnary(0, length_sentence, rootIdx));
            bestParse = new Tree<String>("ROOT", treeList);
        }

        double now = System.currentTimeMillis();
        System.out.println((now-then) + " milliseconds elapsed for sentence length " + length_sentence);
        return MyTreeAnnotations.unAnnotateTree(bestParse);
//        return bestParse; // Return this instead to see what annotated trees look like
    }

    private Tree<String> unaryPathToTree(List<Integer> path, Tree<String> tree, String terminal){
        /**
         * Recursive function that returns connected unary trees, given a path of unary rules (usually from a unary closure)
         * Ends in a terminal (the ultimate word we're trying to reach)
         */
        if (path.size()==1){
            Tree<String> termTree = new Tree<String>(terminal);
            List<Tree<String>> termTreeList = Arrays.asList(termTree);
            tree.setLabel(labelIndexer.get(path.get(0)));
            tree.setChildren(termTreeList);
            return tree;
        } else {
            Tree<String> tempTree = new Tree<String>("Junk");
            tree.setLabel(labelIndexer.get(path.get(0)));
            tempTree = unaryPathToTree(path.subList(1, path.size()), tempTree, terminal);
            List<Tree<String>> tempTreeList = Arrays.asList(tempTree);
            tree.setChildren(tempTreeList);
            return tree;
        }
    }

    Tree<String> getBestTreeUnary(int i, int j, int parent){
        /**
         * Start the top-down reconstruction by calling this on the ROOT index within the unary chart (ie, i=0, j=sentence_length, parent=0)
         * Will share a mutual recursion with getBestTreeBinary, each calling the other.
         *
         * @author Jake Daly
         * @param i, j, parent
         * @return Tree (best scoring unary tree)
         */

        if (debug) {
            System.out.print("getBestTreeUnary... ");
            System.out.println(i + " " + j + " " + parent);
        }
        if (j - i == 1) {
            String word = currentSentence.get(i);
            UnaryRule unaryToPreterminal = new UnaryRule(parent, unaryBP[i][j][parent].child);
            List<Integer> bestPath = UC.getPath(unaryToPreterminal);
            Tree<String> closureTree = new Tree<String>("Junk");
            closureTree = unaryPathToTree(bestPath, closureTree, word);

            return closureTree;
        } else {
            Tree<String> childTree = getBestTreeBinary(i, j, unaryBP[i][j][parent].child);
            return childTree;
        }
    }

    Tree<String> getBestTreeBinary (int i, int j, int parent){
        /**
         * Binary counterpart to getBestTreeUnary. Whenever children = 2, this function should get called.
         * Note the recursion will never end with this function because we will always terminate with unary chart
         *
         * @author Jake Daly
         * @param i, j, parent
         * @return Tree (best scoring binary tree)
         */

        if (debug) {
            System.out.print("getBestTreeBinary... ");
            System.out.println(i + " " + j + " " + parent);
        }
        String label = labelIndexer.get(parent);
        Tree<String> childLeft = getBestTreeUnary(i, binaryBP[i][j][parent].k, binaryBP[i][j][parent].lchild);
        Tree<String> childRight = getBestTreeUnary(binaryBP[i][j][parent].k, j, binaryBP[i][j][parent].rchild);
        List<Tree<String>> childrenList = Arrays.asList(childLeft, childRight);
        Tree<String> bestTree = new Tree<>(label, childrenList);
        return bestTree;
    }

    class BBP {
        /**
         * Binary back pointer (BBP) class
         *
         * @author Jake Daly
         * @param p (parent index number), lc (left child index number), rc (right child index number), split (k)
         */
        int parent;
        int lchild;
        int rchild;
        int k;
        public BBP(int p, int lc, int rc, int split){
            parent = p;
            lchild = lc;
            rchild = rc;
            k = split;
        }
    }

    class UBP {
        /**
         * Unary back pointer (UBP) class
         *
         * @author Jake Daly
         * @param p (parent index number), c (child index number)
         */
        int parent;
        int child;
        public UBP(int p, int c){
            parent = p;
            child = c;
        }
    }

    void doubleFill(double[][][] A, double value){
        for (double[][] outer: A){
            for (double[] inner: outer){
                Arrays.fill(inner, value);
            }
        }
    }

    void initializeUnaryBP(UBP[][][] UnaryBPArray){
        UBP filler = new UBP(-1, -1);
        for (UBP[][] outer : UnaryBPArray){
            for (UBP[] inner : outer){
                Arrays.fill(inner, filler);
            }
        }
    }
    void initializeBinaryBP(BBP[][][] BinaryBPArray){
        BBP filler = new BBP(-1, -1, -1, -1);
        for (BBP[][] outer: BinaryBPArray){
            for (BBP[] inner : outer){
                Arrays.fill(inner, filler);
            }
        }
    }

    private List<Tree<String>> annotateTrees(List<Tree<String>> trees) {
        List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
        for (Tree<String> tree : trees) {
            annotatedTrees.add(MyTreeAnnotations.annotateTreeBinarization(tree));
        }
        return annotatedTrees;
    }

    private int tallySpans(Tree<String> tree, int start) {
        if (tree.isLeaf() || tree.isPreTerminal()) return 1;
        int end = start;
        for (Tree<String> child : tree.getChildren()) {
            int childSpan = tallySpans(child, end);
            end += childSpan;
        }
        String category = tree.getLabel();
        if (!category.equals("ROOT")) spanToCategories.incrementCount(end - start, category, 1.0);
        return end - start;
    }
}
