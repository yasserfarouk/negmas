package genius.gui.tree;

import javax.swing.*;

import genius.core.issue.*;
import genius.core.utility.EvaluatorDiscrete;

/**
*
* @author Richard Noorlandt
* 
*/

public class IssueDiscreteValuePanel extends IssueValuePanel {

	private static final long serialVersionUID = 5969631611077257684L;
	
	//Constructors
	public IssueDiscreteValuePanel(NegotiatorTreeTableModel model, IssueDiscrete issue) {
		super(model, issue);
		init(issue);
	}
	
	private void init(IssueDiscrete issue) {
		String values = "";
		for (int i = 0; i < issue.getNumberOfValues(); i++) {
			values = values + issue.getStringValue(i);
			if (model.getUtilitySpace() != null) {
				EvaluatorDiscrete eval = (EvaluatorDiscrete) model.getUtilitySpace().getEvaluator(issue.getNumber());
				try {
					values += " ("+ eval.getEvaluationNotNormalized(issue.getValue(i)) + ")";
				} catch (Exception e) {
				}
			}
			if (i < issue.getNumberOfValues() - 1) {
				values = values + ", ";
			}
		}
		this.add(new JLabel(values));
		this.setToolTipText(values);
	}
	

	
	public void displayValues(Objective node){
		this.removeAll();
		init((IssueDiscrete) node);
	}
}
