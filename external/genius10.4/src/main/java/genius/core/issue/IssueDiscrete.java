package genius.core.issue;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;

import genius.core.xml.SimpleElement;

/**
 * Specific type of issue in which the value of the issue are a set of objects.
 * An example of a discrete issue is the color of car with the values {Red, Blue, Black}.
 * 
 * @author Tim Baarslag & Dmytro Tykhonov
 */
public class IssueDiscrete extends Issue {
		
	/**
	 * 
	 */
	private static final long serialVersionUID = -6220939483738304249L;

	/** List of possible values the issue can have */
	private List<ValueDiscrete> values; 
	
	 // the descriptions for each value
	private HashMap<ValueDiscrete, String> fDesc=new HashMap<ValueDiscrete, String>();
	
	/**
	 * Create a new discrete issue given the name of the issue, its unique ID,
	 * and an array of possible values.
	 * 
	 * @param name of the issue.
	 * @param issueNumber uniqueID of the isue.
	 * @param values which the issue may take.
	 */
	public IssueDiscrete(String name, int issueNumber, String values[]) {
		super(name, issueNumber);		
	    this.values = new ArrayList<ValueDiscrete>();
	    for(int i=0; i< values.length;i++) {
	        this.values.add(new ValueDiscrete(values[i]));
	    }
	}	
	
	/**
	 * Create a new discrete issue given the name of the issue, its unique ID,
	 * its parent, and an array of possible values.
	 * 
	 * @param name of the issue.
	 * @param issueNumber uniqueID of the isue.
	 * @param values which the issue may take.
	 * @param objParent parent objective of the issue.
	 */
	public IssueDiscrete(String name, int issueNumber, String values[], Objective objParent) {
		super(name, issueNumber, objParent);		
		this.values = new ArrayList<ValueDiscrete>();
	    for(int i=0; i< values.length;i++) {
	    	ValueDiscrete v=new ValueDiscrete(values[i]);
	        this.values.add(v);
	    }
	}
	
	/**
	 * Create a new discrete issue given the name of the issue, its unique ID,
	 * its parent, and an array of possible values and a description for each value.
	 * 
	 * @param name of the issue.
	 * @param issueNumber uniqueID of the isue.
	 * @param values which the issue may take.
	 * @param descriptions array with a description for each value.
	 * @param objParent parent of the issue.
	 */
	public IssueDiscrete(String name, int issueNumber, String values[], String descriptions[],Objective objParent) {
		super(name, issueNumber, objParent);		
		this.values = new ArrayList<ValueDiscrete>();
	    for(int i=0; i<values.length;i++) {
			ValueDiscrete v=new ValueDiscrete(values[i]);
	        this.values.add(v);
	        if (descriptions!=null && descriptions[i]!=null) fDesc.put(v,descriptions[i]);
	    }
	}
	
	public boolean equals(Object o)
	{
		if (!(o instanceof IssueDiscrete)) return false;
		if (!equalContents((Objective)o)) return false; // check the basic things like name
		 // NOTE, we use .equals on issueValues which is an ArrayList.
		 // therefore the ORDER of the issues is critical as well (as it should)
		return values.equals( ((IssueDiscrete)o).getValues());
	}
	
	/**
	 * @return amount of values.
	 */
	public int getNumberOfValues() {
	    return values.size();
	}
		
	/**
	 * Get value by its String representation, <b>null</b> otherwise.
	 * @param index of the value to be returned.
	 * @return value with the given index in the array of values for this issue.
	 */
	public ValueDiscrete getValue(int index) {
		return (ValueDiscrete)values.get(index);
	}
	
	/**
	 * @param index of the value.
	 * @return string of the value, for example "Red".
	 */
	public String getStringValue(int index) {
		return ((ValueDiscrete)values.get(index)).getValue();
	}
	    
	/** 
	 * @param value that is supposed to be one of the alternatives of this issue.
	 * @return index holding that value, or -1 if value is not one of the alternatives.
	 */
	public int getValueIndex(String value) {
	    for(int i=0;i<values.size();i++)
	        if(values.get(i).toString().equals(value)) {
	            return i;
	        }
	    return -1;
	}
	
	/** 
	 * @param value that is supposed to be one of the alternatives of this issue.
	 * @return index holding that value, or -1 if value is not one of the alternatives.
	 */
	public int getValueIndex(ValueDiscrete value) {
	    for(int i=0;i<values.size();i++)
	        if(values.get(i).equals(value)) {
	            return i;
	        }
	    return -1;
	}
	
	/**
	 * Removes all values from this Issue.
	 */
	public void clear(){
		values.clear();
	}
	
	/**
	 * Adds a value.
	 * @param valname The name of the value to add.
	 */
	public void addValue(String valname){
		values.add(new ValueDiscrete(valname));
	}
	
	/**
	 * Adds values.
	 * @param valnames Array with names of values to add.
	 */
	public void addValues(String[] valnames){
		for(int ind=0; ind < valnames.length; ind++){
			values.add(new ValueDiscrete(valnames[ind]));
		}
	}
	
	public boolean checkInRange(Value value) {
			return (getValueIndex(((ValueDiscrete)value).getValue())!=-1);
	}
	
	/**
	 * Gives an enumeration over all values in this discrete issue.
	 * @return An enumeration containing <code>valueDiscrete</code>
	 */
	public List<ValueDiscrete> getValues() {
		return values;
	}
	
	/**
	 * Returns a SimpleElement representation of this issue.
	 * @return The SimpleElement with this issues attributes
	 */
	public SimpleElement toXML(){
		SimpleElement thisIssue = new SimpleElement("issue");
		thisIssue.setAttribute("name", getName());
		thisIssue.setAttribute("index", ""+getNumber());
		thisIssue.setAttribute("etype", "discrete");
		thisIssue.setAttribute("type", "discrete");
		thisIssue.setAttribute("vtype", "discrete");
		//TODO find some way of putting the items in. Probably in much the same way as weights.
		for(int item_ind = 0; item_ind < values.size(); item_ind++){
			SimpleElement thisItem = new SimpleElement("item");
			thisItem.setAttribute("index", "" + (item_ind +1)); //One off error?
			thisItem.setAttribute("value", values.get(item_ind).toString());
			String desc=fDesc.get(values.get(item_ind));
			if (desc!=null) thisItem.setAttribute("description", desc);
			thisIssue.addChildElement(thisItem);
 		}
		return thisIssue;
		
	}

	/**
	 * Sets the desc for value <code>val</code>. If the value doesn't exist yet in this Evaluator,
	 * add it as well.
	 * @param val The value to have it's desc set/modified
	 * @param desc The new desc of the value.
	 */
	public void setDesc(ValueDiscrete val, String desc)
	{
		fDesc.put(val, desc);
	}
	
	/**
	 * @param value
	 * @return description of the given value. A description is an optional explanation of the value.
	 */
	public String getDesc(ValueDiscrete value)
	{ return fDesc.get(value); }


	@Override
	public ISSUETYPE getType() {
		return ISSUETYPE.DISCRETE;
	}

	@Override
	public String convertToString() {
		return "discrete";
	}
}