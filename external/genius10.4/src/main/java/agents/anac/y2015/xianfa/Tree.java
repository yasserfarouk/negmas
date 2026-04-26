package agents.anac.y2015.xianfa;

import java.util.ArrayList;

public class Tree {
	private Node root;
	//private int depth=0;
	private ArrayList<ArrayList<Node>> nodes = new ArrayList<ArrayList<Node>>();
	boolean setRoot = false;
	
	public Tree() {
		
	}
	
	public void addNewDepth() {
		if (!setRoot) {
			setRoot = true;
			root = new Node();
			ArrayList<Node> lvl1 = new ArrayList<Node>();
			lvl1.add(root);
			nodes.add(lvl1);
		} else {
			ArrayList<Node> newLvl = new ArrayList<Node>();
			nodes.add(newLvl);
		}
	}
	
	public void addNodeInDepth(Node node, int depth) {
		nodes.get(depth).add(node);
	}
	
	public void setRoot(Node node) {
		root = new Node();
		ArrayList<Node> lvl1 = new ArrayList<Node>();
		lvl1.add(root);
		nodes.add(lvl1);
	}
	
	public Node getRoot() {
		return root;
	}
	
	public void addNode(Node node) {
		
	}
	
	public int getSizeOfLevel(int depth) {
		return nodes.get(depth).size();
	}
	
	public Node getMemberInLevel(int depth, int member) {
		return nodes.get(depth).get(member);
	}
	
	public int getLevels() {
		return nodes.size();
	}

}
