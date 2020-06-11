package edu.berkeley.nlp.assignments.parsing.student;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.Filter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MyTreeAnnotations {

    /**
     * Class for annotating and de-annotating trees.
     * Horizontal and vertical Markovization are used, each of order 2
     *
     * @author Jake Daly
     * @param unAnnotatedTree
     * @return Tree
     */
    public static Tree<String> annotateTreeBinarization(Tree<String> unAnnotatedTree) {

        return binarizeTree(unAnnotatedTree);
    }

    private static Tree<String> binarizeTree(Tree<String> tree) {
        /**
         * Binarizes and annotates the given unannotated tree, returns an annotated tree. Recursive function
         *
         * @augmented Jake Daly
         * @param tree
         * @return tree
         */

        String label = tree.getLabel();
        if (tree.isLeaf()) return new Tree<String>(label);
        String parent = label.split("\\^")[0].replace("@", "").trim();
        String child;

        // Apply vertical Markov
        for(Tree<String> c : tree.getChildren()) {
            if(c.isPreTerminal() || c.isPhrasal()) {
                child = c.getLabel().concat("^" + parent);
                c.setLabel(child);
            }
        }

        if (tree.getChildren().size() == 1) { return new Tree<String>(label, Collections.singletonList(binarizeTree(tree.getChildren().get(0)))); }
        
        // if tree is length of list(children) is > 1, recurse!
        String intermediateLabel = "@" + label + "->...";
        Tree<String> intermediateTree = binarizeTreeHelper(tree, 0, intermediateLabel);
        return new Tree<String>(label, intermediateTree.getChildren());
    }

    private static Tree<String> binarizeTreeHelper(Tree<String> tree, int numChildrenGenerated, String intermediateLabel) {
        /**
         * Use when number of children is 2 or greater. Will recursively call itself on each of the children.
         * This is where horizontal Markovization occurs.
         *
         * @augmented Jake Daly
         * @param tree, numChildrenGenerated, intermediateLabel
         * @return tree
         */

        Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
        List<Tree<String>> children = new ArrayList<Tree<String>>();
        children.add(binarizeTree(leftTree));

        if (numChildrenGenerated < tree.getChildren().size() - 1) {
            String label = "";
            String sib = leftTree.getLabel().split("\\^")[0].trim();
            if(intermediateLabel.indexOf("...") != -1) {
                String[] splits = intermediateLabel.split("\\.\\.\\.");
                if(splits.length > 2) { label = splits[0] + "...[" + splits[2] + "][" + sib + "]"; }
                else {
                    if(splits.length == 1) { label = intermediateLabel.concat(sib); }
                    else { label = intermediateLabel.concat("...[" + sib + "]"); }
                }
            }
            Tree<String> rightTree = binarizeTreeHelper(tree, numChildrenGenerated + 1, label);
            children.add(rightTree);
        }
        return new Tree<String>(intermediateLabel, children);
    }

    public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {
        // Remove intermediate nodes (labels beginning with "@"
        // Remove all material on node labels which follow their base symbol (cuts anything after <,>,^,=,_ or ->)
        // Examples: a node with label @NP->DT_JJ will be spliced out, and a node with label NP^S will be reduced to NP
        Tree<String> debinarizedTree = Trees.spliceNodes(annotatedTree, new Filter<String>()
        {
            public boolean accept(String s) {
                return s.startsWith("@");
            }
        });
        Tree<String> unAnnotatedTree = (new Trees.LabelNormalizer()).transformTree(debinarizedTree);
        return unAnnotatedTree;
    }

}
