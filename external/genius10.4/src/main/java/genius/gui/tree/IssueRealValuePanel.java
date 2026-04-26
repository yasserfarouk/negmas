package genius.gui.tree;

import javax.swing.*;

import genius.core.issue.*;

/**
*
* @author Richard Noorlandt
* 
*/

public class IssueRealValuePanel extends IssueValuePanel {

	//Attributes
	private static final long serialVersionUID = 28688258478356130L;

	//Constructors
	public IssueRealValuePanel(NegotiatorTreeTableModel model, IssueReal issue) {
		super(model, issue);
		
		init(issue);
	}
	
	//Methods
	private void init(IssueReal issue) {
		this.add(new JLabel("Min: " + issue.getLowerBound() + "\tMax: " + issue.getUpperBound()));
	}
	
	public void displayValues(Objective node){
		this.removeAll();
		init(((IssueReal)node));
	}
}
