package agents.anac.y2015.xianfa;

import java.util.ArrayList;

import genius.core.issue.Value;

public class Node {
	
	private Node parent;
	private ArrayList<Node> child = new ArrayList<Node>();
	private int depth;
	private int number;
	private Value value;
	private double evaluation;
	private boolean visited = false;
	
	public boolean isVisited() {
		return visited;
	}

	public void setVisited(boolean visited) {
		this.visited = visited;
	}

	public Node() {
		
	}
	
	public Node(int number, Value value, int depth, double evaluation) {
		this.number = number;
		this.depth = depth;
		this.value = value;
		this.evaluation = evaluation;
	}
	
	public void add(Node node) {
		child.add(node);
		node.setParent(this);
	}
	
	public Node getParent() {
		return parent;
	}

	public ArrayList<Node> getChild() {
		return child;
	}

	public int getDepth() {
		return depth;
	}

	public int getNumber() {
		return number;
	}

	public Value getValue() {
		return value;
	}

	public double getEvaluation() {
		return evaluation;
	}

	public void setParent(Node node) {
		parent = node;
	}
	
	public void show() {
		System.out.println("this is " + number + " & " + depth + " & " + value + " & " + evaluation);
	}

}
