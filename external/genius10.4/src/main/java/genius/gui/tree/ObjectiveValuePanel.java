package genius.gui.tree;

import javax.swing.*;

import genius.core.issue.*;

/**
*
* @author Richard Noorlandt
* 
*/

public class ObjectiveValuePanel extends IssueValuePanel {

	private static final long serialVersionUID = -4899545167184650567L;

	//Constructors
	public ObjectiveValuePanel(NegotiatorTreeTableModel model, Objective objective) {
		super(model, objective);
		
		init(objective);
	}
	
	//Methods
	private void init(Objective objective) {
		this.add(new JLabel("This == Objective"));
	}
	
	public void displayValues(Objective node){
		this.removeAll();
		init(node);
	}
}
