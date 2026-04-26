package genius.gui.dialogs;

import java.awt.CardLayout;
import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.JOptionPane;

import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;
import genius.gui.tree.NegotiatorTreeTableModel;
import genius.gui.tree.DomainAndProfileEditorPanel;

/**
 * 
 * @author Richard Noorlandt
 *
 *         This launches a editissue dialog window. Wouter: this is ugly. The
 *         EditIssueDialog also handles editing of evaluators. it gets access to
 *         the util space via the treeFrame, the parent of this dialog.
 */
public class EditIssueDialog extends NewIssueDialog {

	private static final long serialVersionUID = 5730169200768833303L;
	private Issue issue;

	// Constructors
	public EditIssueDialog(DomainAndProfileEditorPanel owner, Issue issue) {
		this(owner, false, issue);
	}

	public EditIssueDialog(DomainAndProfileEditorPanel owner, boolean modal, Issue issue) {
		this(owner, modal, "Edit Issue", issue);
		this.issue = issue;
	}

	public EditIssueDialog(DomainAndProfileEditorPanel owner, boolean modal, String name, Issue issue) {
		super(owner, modal, name);
		this.issue = issue;
		setPanelContents(issue);
	}

	/**
	 * Load the appropriate contents into the right panel.
	 * 
	 * @param issue
	 */
	private void setPanelContents(Issue issue) {
		AdditiveUtilitySpace utilSpace = treeFrame.getNegotiatorTreeTableModel().getUtilitySpace();

		nameField.setText(issue.getName());
		numberField.setText("" + issue.getNumber());

		if (!treeFrame.isDomain() || !treeFrame.hasNoProfiles()) {
			nameField.setEnabled(false);
		}

		if (issue instanceof IssueDiscrete) {
			this.issueType.setSelectedItem(DISCRETE);
			this.issueType.setEnabled(false);
			((CardLayout) issuePropertyCards.getLayout()).show(issuePropertyCards, DISCRETE);
			List<ValueDiscrete> values = ((IssueDiscrete) issue).getValues();

			String valueString = "";
			String descString = "";
			for (ValueDiscrete val : values) {
				valueString = valueString + val.getValue() + "\n";
				String desc = ((IssueDiscrete) issue).getDesc(val);
				if (desc != null)
					descString = descString + desc;
				descString = descString + "\n";
			}
			discreteTextArea.setText(valueString);
			if (utilSpace != null) {
				EvaluatorDiscrete eval = (EvaluatorDiscrete) utilSpace.getEvaluator(issue.getNumber());
				if (eval != null) {
					// load the eval values
					valueString = "";

					for (ValueDiscrete val : values) {
						Integer util = eval.getValue(val); // get the utility
															// for this value
						// System.out.println("util="+util);
						if (util != null)
							valueString = valueString + util;

						valueString = valueString + "\n";
					}
					discreteTextEvaluationArea.setText(valueString);
				}
			}
		} else if (issue instanceof IssueInteger) {
			this.issueType.setSelectedItem(INTEGER);
			this.issueType.setEnabled(false);

			((CardLayout) issuePropertyCards.getLayout()).show(issuePropertyCards, INTEGER);
			integerMinField.setText("" + ((IssueInteger) issue).getLowerBound());
			integerMaxField.setText("" + ((IssueInteger) issue).getUpperBound());
			if (utilSpace != null) {
				EvaluatorInteger eval = (EvaluatorInteger) utilSpace.getEvaluator(issue.getNumber());

				if (eval != null) {
					integerUtilityLowestValue.setText("" + eval.getUtilLowestValue());
					integerUtilityHighestValue.setText("" + eval.getUtilHighestValue());
				}
			}
		} else if (issue instanceof IssueReal) {
			this.issueType.setSelectedItem(REAL);
			this.issueType.setEnabled(false);
			((CardLayout) issuePropertyCards.getLayout()).show(issuePropertyCards, REAL);
			realMinField.setText("" + ((IssueReal) issue).getLowerBound());
			realMaxField.setText("" + ((IssueReal) issue).getUpperBound());
			if (utilSpace != null) {
				EvaluatorReal eval = (EvaluatorReal) utilSpace.getEvaluator(issue.getNumber());
				if (eval != null) {
					switch (eval.getFuncType()) {
					case LINEAR:
						realLinearField.setText("" + eval.getLinearParam());
					case CONSTANT:
						realParameterField.setText("" + eval.getConstantParam());
					default:
						break;
					}
				}
			}
		}
	}

	/**
	 * Overrides getObjectiveNumber from NewObjectiveDialog
	 */
	@Override
	protected int getObjectiveNumber() throws InvalidInputException {
		try {
			return Integer.parseInt(numberField.getText());
		} catch (Exception e) {
			throw new InvalidInputException("Error reading objective number from (hidden) field.");
		}
	}

	/**
	 * Overrides actionPerformed from NewIssueDialog.
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == okButton) {
			if (issue == null)
				return;

			boolean valid = true;
			if (issue instanceof IssueInteger) {
				if (((NegotiatorTreeTableModel) treeFrame.getTreeTable().getTree().getModel())
						.getUtilitySpace() != null) {
					double utilLIV = Double.parseDouble(integerUtilityLowestValue.getText());
					double utilHIV = Double.parseDouble(integerUtilityHighestValue.getText());
					if (utilLIV < 0.0 || utilLIV > 1.0) {
						valid = false;
						JOptionPane.showConfirmDialog(null,
								"The utility of the lowest value should be \n" + "in the range [0, 1]", "Input",
								JOptionPane.PLAIN_MESSAGE);
					} else if (utilHIV < 0.0 || utilHIV > 1.0) {
						valid = false;
						JOptionPane.showConfirmDialog(null,
								"The utility of the heighest value should be \n" + "in the range [0, 1]", "Input",
								JOptionPane.PLAIN_MESSAGE);
					}
				}
			}
			if (valid) {
				updateIssue(issue);

				// Notify the model that the contents of the treetable have
				// changed
				NegotiatorTreeTableModel model = (NegotiatorTreeTableModel) treeFrame.getTreeTable().getTree()
						.getModel();

				(model.getIssueValuePanel(issue)).displayValues(issue);
				Object[] path = { model.getRoot() };
				if (treeFrame.getTreeTable().getTree().getSelectionPath() != null) {
					path = treeFrame.getTreeTable().getTree().getSelectionPath().getPath();
				}
				model.treeStructureChanged(this, path);

				this.dispose();
			}
		} else if (e.getSource() == cancelButton) {
			this.dispose();
		}
	}
}