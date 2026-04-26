package genius.core.issue;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import javax.swing.tree.MutableTreeNode;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

import genius.core.xml.SimpleElement;

/**
 * Objective is superclass of issues and can be configured in a parent-child
 * tree-like construction. Some work needs to be done to guarantee consistency
 * of the tree. Methods like setParent don't signal the parent that it has a new
 * child.
 * 
 * @author Richard Noorlandt, W. Pasman
 */
public class Objective implements MutableTreeNode, Serializable {

	private static final long serialVersionUID = -2929020937103860315L;
	private int number;
	private String name;
	private String description = "";
	private Object userObject; // can be a picture, for instance
	private Objective parent; // Wouter: null if no parent available?
	private ArrayList<Objective> children = new ArrayList<Objective>();

	public Objective() {
	}

	public Objective(Objective parent) {
		this.parent = parent;
		this.name = "No name specified";
	}

	public Objective(Objective parent, String name, int nr) {
		this.parent = parent;
		this.name = name;
		this.number = nr;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((description == null) ? 0 : description.hashCode());
		result = prime * result + ((name == null) ? 0 : name.hashCode());
		result = prime * result + number;
		return result;
	}

	/**
	 * @return true if given object is an Objective and number, name, and
	 *         children are the same we don't care about the description and
	 *         user objects.
	 */
	public boolean equals(Object o) {
		if (!(o instanceof Objective))
			return false;
		return equalContents((Objective) o);
	}

	/**
	 * check the contents but don't check for the class type anymore.
	 * 
	 * @param obj
	 *            the objective to be compared
	 * @return true if number, name, and children are the same we don't care
	 *         about the description and user objects.
	 */
	public boolean equalContents(Objective obj) {
		if (number != obj.getNumber() || (!name.equals(obj.getName())))
			return false;

		for (Objective child : children)
			if (!(child.equals(obj.getChildWithID(child.getNumber()))))
				return false;
		return true;

	}

	/**
	 * @return the name of this node.
	 */
	public String getName() {
		return name;
	}

	/**
	 * Sets a new name for this node.
	 * 
	 * @param newName
	 *            the new name for this node.
	 */
	public void setName(String newName) {
		name = newName;
	}

	/**
	 * @return the number of this Objective / Issue.
	 */
	public int getNumber() {
		return number;
	}

	/**
	 * Sets the number of this Objective / Issue.
	 */
	public void setNumber(int nr) {
		number = nr;
	}

	/**
	 * @return this node's description.
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * Sets a new description for this node.
	 * 
	 * @param newDescription
	 *            the new description.
	 */
	public void setDescription(String newDescription) {
		description = newDescription;
	}

	/**
	 * @return the user object containted within this node.
	 */
	public Object getUserObject() {
		return userObject;
	}

	/**
	 * @return true if and only if this node is of the Objective type, but not
	 *         of the Issue type. This is implemented by looking if the receiver
	 *         is instanceof Objective, but not instanceof Issue.
	 */
	public boolean isObjective() {
		return ((this instanceof Objective) && !(this instanceof Issue));
	}

	/**
	 * @return true if and only if this node is of the Issue type. It is
	 *         implemented with instanceof Issue.
	 */
	public boolean isIssue() {
		return (this instanceof Issue);
	}

	/**
	 * This method is added for convenience. It is simply an alternative
	 * implementation of the getType method from the Issue class. It will always
	 * return that the type is Objective. This method must be overridden in
	 * Issue to return the type of the Issue.
	 * 
	 * @return ISSUETYPE.OBJECTIVE
	 */
	public ISSUETYPE getType() {
		return ISSUETYPE.OBJECTIVE;
	}

	/**
	 * Adds a child to this Objective. The new child must be an Objective or an
	 * issue. The child is messaged to set it's parent to the receiver.
	 * 
	 * @param newObjective
	 *            a child to be added to this Objective.
	 */
	public void addChild(Objective newObjective) {
		children.add(newObjective);
		newObjective.setParent(this);
	}

	/**
	 * This method does a recursive depth-first search on the subtree that is
	 * rooted at the receiver, and returns the first Objective or Issue with the
	 * given number. If there is no matching node found, null is returned.
	 * 
	 * @param objectiveNr
	 *            the Objective/Issue number that is being searched for.
	 * @return the Objective/Issue with the given number, or null if the
	 *         requested Objective/Issue is not found in the subtree that is
	 *         rooted at the receiver.
	 */
	public Objective getObjective(int objectiveNr) {
		if (getNumber() == objectiveNr)
			return this;

		Enumeration<Objective> descendants = children();
		while (descendants.hasMoreElements()) {
			Objective obj = descendants.nextElement().getObjective(objectiveNr);
			if (obj != null)
				return obj;
		}
		return null;
	}

	/**
	 * @return the name of this Objective.
	 */
	public String toString() {
		return name;
	}

	/**
	 * 
	 * @return an Objective[] containing this node's siblings, or an empty array
	 *         if this Objective's parent is null.
	 */
	public Objective[] getSiblings() {
		Objective[] siblings;
		Objective parent = getParent();
		if (parent == null) {
			return new Objective[0];
		}

		siblings = new Objective[parent.getChildCount() - 1];
		// Fill the array with siblings. i is used for enumerating all children
		// of the parent, j is used to index
		// the array. j is needed because there is one less sibling than there
		// are children.
		for (int i = 0, j = 0; i < siblings.length; i++, j++) {
			Objective obj = parent.getChildAt(i);
			if (obj == this) {
				// Don't add to the array, decrease j to undo j++
				j--;
			} else {
				siblings[j] = obj;
			}
		}

		return siblings;
	}

	/**
	 * Check whether or not a particular Objective is a child of the receiver.
	 * Comparison is based on simple pointer comparison.
	 * 
	 * @param child
	 *            the potential child of the callee.
	 * @return true iff child is a direct child of the receiving node.
	 */
	public boolean isParent(Objective child) {
		boolean result = false;
		Enumeration<Objective> childEnum = children();
		while (childEnum.hasMoreElements()) {
			Objective obj = childEnum.nextElement();
			if (obj == child) {
				result = true;
			}
		}
		return result;
	}

	// implements TreeNode interface

	/**
	 * @return an Enumeration of this Objective's children.
	 */
	public Enumeration<Objective> children() {
		return Collections.enumeration(children);
	}

	/**
	 * Wouter: added bcause I dont have time to change all Vector and
	 * Enumerators to ArrayList code
	 */
	public ArrayList<Objective> getChildren() {
		return new ArrayList<Objective>(children);
	}

	/**
	 * @return true iff the node is an OBJECTIVE, of false if the node is an
	 *         ISSUE.
	 */
	public boolean getAllowsChildren() {
		return (this instanceof Objective);
	}

	/**
	 * @return the child at the given index, or null if the index is invalid.
	 */
	public Objective getChildAt(int childIndex) {
		if (childIndex < children.size() && childIndex >= 0)
			return children.get(childIndex);
		else
			return null;
	}

	/**
	 * @param ID
	 *            is the ID number of the needed child
	 * @return Objective, or null.
	 */
	public Objective getChildWithID(int ID) {
		for (Objective obj : getChildren())
			if (obj.getNumber() == ID)
				return obj;
		return null;
	}

	/**
	 * @return the number of children of this node.
	 */
	public int getChildCount() {
		return children.size();
	}

	/**
	 * @return the index of node in the receivers children. If the receiver does
	 *         not contain node, -1 will be returned.
	 */
	public int getIndex(TreeNode node) {
		for (int i = 0; i < children.size(); i++) {
			if (node == children.get(i))
				return i;
		}
		return -1;
	}

	/**
	 * @return the parent Objective of the receiver.
	 */
	public Objective getParent() {
		return parent;
	}

	/**
	 * @return is the receiving node is a leaf node. A Objective is a leaf node
	 *         when it is of the ISSUE type.
	 */
	public boolean isLeaf() {
		return isIssue();
	}

	/**
	 * This method recursively calculates the highest Objective / Issue number
	 * in the subtree rooted at the callee.
	 * 
	 * @param lowerBound
	 *            the number to be returned must have at least this value. Used
	 *            for the recursive implementation.
	 * @return the highest Objective number within this subtree that is greater
	 *         than lowerBound, or otherwise lowerBound.
	 */
	public int getHighestObjectiveNr(int lowerBound) {
		if (getNumber() > lowerBound)
			lowerBound = getNumber();

		Enumeration<Objective> descendants = children();
		while (descendants.hasMoreElements()) {
			Objective obj = descendants.nextElement();
			if (obj.getNumber() > lowerBound)
				lowerBound = obj.getNumber();
			lowerBound = obj.getHighestObjectiveNr(lowerBound);
		}

		return lowerBound;
	}

	// implements MutableTreeNode interface

	/**
	 * Adds child to the receiver at index. child will be messaged with
	 * setParent. Nodes at the given index and above are moved one place up to
	 * make room for the new node. If index > getChildCount() or index < 0,
	 * nothing happens.
	 * 
	 * @param child
	 *            the Objective to be inserted. If child is no NegotionTreeNode,
	 *            a ClassCastException will be thrown.
	 * @param index
	 *            the index where the new node is to be inserted.
	 */
	public void insert(MutableTreeNode child, int index) {
		if (index <= getChildCount() && index >= 0) {
			children.add(index, (Objective) child);
			child.setParent(this);
		}
	}

	/**
	 * Removes the child at the given index, setting it's parent to null. If
	 * index >= getChildCount or index < 0, nothing happens.
	 */
	public void remove(int index) {
		if (index < getChildCount() && index >= 0) {
			getChildAt(index).setParent(null);
			children.remove(index);
		}
	}

	/**
	 * Removes node from the receiver's children, and sets it's parent to null.
	 * If node is not one of the receiver's children, nothing happens.
	 */
	public void remove(MutableTreeNode node) {
		for (int i = 0; i < children.size(); i++) {
			if (node == children.get(i)) {
				getChildAt(i).setParent(null);
				children.remove(i);
			}
		}
	}

	/**
	 * Removes the subtree rooted at this node from the tree, giving this node a
	 * null parent. Does nothing if this node is the root of its tree.
	 */
	public void removeFromParent() {
		if (parent != null) {
			parent.remove(this);
			parent = null;
		}
	}

	/**
	 * Sets this node's parent to newParent but does not change the parent's
	 * child array. This method is called from insert() and remove() to reassign
	 * a child's parent, it should not be messaged from anywhere else. Also,
	 * newParent is cast to a Objective. Calling this method with a different
	 * type of TreeNode will result in a ClassCastException.
	 */
	public void setParent(MutableTreeNode newParent) {
		parent = (Objective) newParent;
	}

	/**
	 * Sets a user object associated with the receiving Objective. This method
	 * is primarily available in order to implement the MutableTreeNode
	 * interface, but because the user object can be of any type it may well be
	 * used to associate extra information about the node. For instance a
	 * picture of this node's OBJECTIVE or ISSUE.
	 */
	public void setUserObject(Object object) {
		userObject = object;
	}

	// Enumeration methods

	/**
	 * Constructs an Enumeration of the entire subtree of the receiver
	 * (including itself) in preorder. The enumeration is immediately
	 * constructed against the current state of the tree, so modifications to
	 * the tree afterwards are not reflected in the Enumeration.
	 * 
	 * @return the preorder Enumeration of the subtree.
	 */
	public Enumeration<Objective> getPreorderEnumeration() {
		return getPreorderElements(true, true, new Vector<Objective>()).elements();
	}

	/**
	 * Constructs an Enumeration of the entire subtree of the receiver
	 * (including itself) in preorder, containting only the Ojectives, but not
	 * the Issues.. The enumeration is immediately constructed against the
	 * current state of the tree, so modifications to the tree afterwards are
	 * not reflected in the Enumeration.
	 * 
	 * @return the preorder Enumeration of Objectives from the subtree.
	 */
	public Enumeration<Objective> getPreorderObjectiveEnumeration() {
		return getPreorderElements(true, false, new Vector<Objective>()).elements();
	}

	/**
	 * Constructs an Enumeration of the entire subtree of the receiver
	 * (including itself) in preorder, containing only the Issues, but not the
	 * normal Objectives. The enumeration is immediately constructed against the
	 * current state of the tree, so modifications to the tree afterwards are
	 * not reflected in the Enumeration.
	 * 
	 * @return the preorder Enumeration of Issues from the subtree.
	 */
	public Enumeration<Objective> getPreorderIssueEnumeration() {
		return getPreorderElements(false, true, new Vector<Objective>()).elements();
	}

	/**
	 * Makes a preorder traversal of the tree, adding Objectives and Issues to
	 * the Vector elems.
	 * 
	 * @param includeObjectives
	 *            defines if Objectives should be included in the Vector.
	 * @param includeIssues
	 *            defines if Issues should be included in the Vector.
	 * @param elems
	 *            the Vector that will contain all the elements of the tree. It
	 *            may not be null.
	 * @return elems, with all descendants added to the vector in preorder.
	 */
	private Vector<Objective> getPreorderElements(boolean includeObjectives, boolean includeIssues,
			Vector<Objective> elems) {
		if (isIssue() && includeIssues) {
			elems.add(this);
			// If this is an Issue, it doesn't have children so we can return.
			return elems;
		} else if (isObjective()) {
			if (includeObjectives)
				elems.add(this);

			Enumeration<Objective> desc = children();
			while (desc.hasMoreElements()) {
				desc.nextElement().getPreorderElements(includeObjectives, includeIssues, elems);
			}
		}
		return elems;
	}

	/**
	 * Returns an xml representation of this Objective and all Objectives and
	 * issues underneath it.
	 */
	public SimpleElement toXML() {
		SimpleElement xmlTree = new SimpleElement("objective");
		xmlTree.setAttribute("name", name);
		xmlTree.setAttribute("index", "" + number);
		xmlTree.setAttribute("description", "" + description);
		xmlTree.setAttribute("type", "objective");
		xmlTree.setAttribute("etype", "objective");
		// Recurse over this object's children.
		Enumeration<Objective> kidsEnum = this.children();
		while (kidsEnum.hasMoreElements()) {
			// how to get the weights? Wouter: domain has no weights!!
			xmlTree.addChildElement((kidsEnum.nextElement()).toXML());
		}
		return xmlTree;
	}

	/**
	 * @return treepath to (and including) this objective. requires that the
	 *         parent fields are set properly and that this implements
	 *         MutableTreeNode.
	 * 
	 */
	public TreePath getPath() {
		if (parent == null)
			return new TreePath((Object) this);
		return parent.getPath().pathByAddingChild((Object) this);
	}
}