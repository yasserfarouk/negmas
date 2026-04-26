package genius.core.issue;

import genius.core.xml.SimpleElement;

/**
 * Class {@link Issue} represents a negotiation issue to be settled in a negotiation. 
 * Issues in a domain are identified by unique <code>index</code> field.
 *
 * @author Tim Baarslag & Dmytro Tykhonov
 */
public abstract class Issue extends Objective {
    
    /**
	 * 
	 */
	private static final long serialVersionUID = -7635200438240075796L;

	/**
     * Creates a new issue give its name and number.
     * @param name of the issue.
     * @param issueNumber unique ID of the issue.
     */
    public Issue(String name, int issueNumber) {
        super(null, name, issueNumber);
    }
    
    /**
     * Creates a new issue give its name, number, and parent.
     * @param name of the issue.
     * @param issueNumber uniqueID of the issue.
     * @param parent objective of the issue.
     */
    public Issue (String name, int issueNumber, Objective parent) {
    	super(parent, name, issueNumber);
    }
    
    public abstract ISSUETYPE getType();
    
    /**
     * @return corresponding string representation
     */
    public abstract String convertToString();
    
    /**
     * Method to check if the given value is in the range specified
     * by the issue.
     * @param value to be checked.
     * @return true if in range.
     */
    public abstract boolean checkInRange(Value value);
	
	/**
	 * Overrides addChild from Objective to do nothing, since Issues can't have children. This
	 * method simply returns without doing anything. 
	 * @param newObjective gets negated.
	 */
	public void addChild(Objective newObjective) { }
	
	/**
	 * Returns a SimpleElement representation of this issue.
	 * @return The SimpleElement with this issues name and index.
	 */
	public SimpleElement toXML(){
		SimpleElement thisIssue = new SimpleElement("issue");
		thisIssue.setAttribute("name", getName());
		thisIssue.setAttribute("index", ""+getNumber());
		return thisIssue;
	}
}