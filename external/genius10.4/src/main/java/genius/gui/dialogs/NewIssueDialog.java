package genius.gui.dialogs;

import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Component;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.ArrayList;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;
import genius.gui.tree.NegotiatorTreeTableModel;
import genius.gui.tree.DomainAndProfileEditorPanel;

/**
 * A dialog window of Genius GUI used to createFrom a new issue and/or evaluator
 * for a issue.
 * 
 * 
 * @author Dmytro Tykhonov
 *
 */
public class NewIssueDialog extends NewObjectiveDialog implements ItemListener {

	private static final long serialVersionUID = 329109532781050011L;
	// Variables
	protected static final String DISCRETE = "Discrete";
	protected static final String INTEGER = "Integer";
	protected static final String REAL = "Real";

	protected JComboBox issueType;
	protected String[] issueTypes;
	protected JPanel issuePropertyCards;
	protected JPanel issuePropertyPanel;
	protected JPanel discretePanel;
	protected JPanel integerPanel;
	protected JPanel realPanel;

	protected JTextArea discreteTextArea;
	protected JTextArea discreteTextEvaluationArea;

	protected JTextField integerMinField;
	protected JTextField integerOtherField;
	protected JTextField integerUtilityLowestValue;
	protected JTextField integerUtilityHighestValue;
	protected JTextField integerMaxField;

	protected JTextField realMinField;
	protected JTextField realOtherField;
	protected JTextField realLinearField;
	protected JTextField realParameterField;
	protected JTextField realMaxField;

	// Constructors
	public NewIssueDialog(DomainAndProfileEditorPanel owner) {
		this(owner, false);
	}

	public NewIssueDialog(DomainAndProfileEditorPanel owner, boolean modal) {
		this(owner, modal, "Create new Issue");
	}

	public NewIssueDialog(DomainAndProfileEditorPanel owner, boolean modal, String name) {
		super(owner, modal, name); // This returns only after user filled in the
									// form and pressed OK
	}

	// Methods
	protected void initPanels() {
		super.initPanels();
		JPanel tmpIssPropP = constructIssuePropertyPanel();

		this.add(tmpIssPropP, BorderLayout.CENTER);

	}

	private JPanel constructIssuePropertyPanel() {
		String[] issueTypesTmp = { DISCRETE, INTEGER };
		issueTypes = issueTypesTmp;

		// Initialize the comboBox.
		issueType = new JComboBox(issueTypes);
		issueType.setSelectedIndex(0);
		issueType.addItemListener(this);

		// Initialize the input components
		discreteTextArea = new JTextArea(20, 10);
		discreteTextEvaluationArea = new JTextArea(20, 4);

		integerMinField = new JTextField(15);
		integerOtherField = new JTextField(15);
		integerUtilityLowestValue = new JTextField(15);
		integerUtilityHighestValue = new JTextField(15);

		integerMaxField = new JTextField(15);
		realMinField = new JTextField(15);
		realOtherField = new JTextField(15);
		realLinearField = new JTextField(15);
		realParameterField = new JTextField(15);
		realMaxField = new JTextField(15);

		// Initialize the panels.
		discretePanel = constructDiscretePanel();
		integerPanel = constructIntegerPanel();
		realPanel = constructRealPanel();

		issuePropertyCards = new JPanel();
		issuePropertyCards.setLayout(new CardLayout());
		issuePropertyCards.add(discretePanel, DISCRETE);
		issuePropertyCards.add(integerPanel, INTEGER);
		issuePropertyCards.add(realPanel, REAL);

		issuePropertyPanel = new JPanel();
		issuePropertyPanel.setBorder(BorderFactory.createTitledBorder("Issue Properties"));
		issuePropertyPanel.setLayout(new BorderLayout());
		issuePropertyPanel.add(issueType, BorderLayout.PAGE_START);
		issuePropertyPanel.add(issuePropertyCards, BorderLayout.CENTER);

		return issuePropertyPanel;
	}

	private JPanel constructDiscretePanel() {
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.LINE_AXIS));

		JPanel textPanel = new JPanel();

		textPanel.setLayout(new BoxLayout(textPanel, BoxLayout.PAGE_AXIS));
		JLabel textLabel = new JLabel("Edit the discrete values below.");
		textPanel.add(textLabel);
		textPanel.add(new JScrollPane(discreteTextArea));
		panel.add(textPanel);

		if (!treeFrame.isDomain()) {
			JPanel evalPanel = new JPanel();
			evalPanel.setLayout(new BoxLayout(evalPanel, BoxLayout.PAGE_AXIS));
			JLabel evalLabel = new JLabel("Evaluation values.");
			evalPanel.add(evalLabel);
			evalPanel.add(new JScrollPane(discreteTextEvaluationArea));
			panel.add(evalPanel);
		}

		// for a domain, do not show the evaluations
		if (treeFrame.isDomain()) {
			discreteTextEvaluationArea.setVisible(false);
		}
		discreteTextArea.setEditable(false);
		if (treeFrame.isDomain() && treeFrame.hasNoProfiles()) { // so it's a
																	// domain
																	// with no
																	// profiles
			discreteTextArea.setEditable(true);
		}
		return panel;
	}

	private JPanel constructIntegerPanel() {
		JPanel panel = new JPanel();

		GridLayout layout = new GridLayout(6, 4);

		panel.setLayout(layout);
		// SPACING
		for (int i = 0; i < 4; i++) {
			panel.add(new JLabel());
		}

		panel.add(new JLabel("Minimum value: "));
		panel.add(integerMinField);

		panel.add(new JLabel("Evaluation of minimum value: "));
		panel.add(integerUtilityLowestValue);

		for (int i = 0; i < 4; i++) {
			panel.add(new JLabel());
		}

		panel.add(new JLabel("Maximum value: "));
		panel.add(integerMaxField);

		panel.add(new JLabel("Evaluation of maximum value: "));
		panel.add(integerUtilityHighestValue);

		if (((NegotiatorTreeTableModel) treeFrame.getTreeTable().getTree().getModel()).getUtilitySpace() == null) {
			integerUtilityLowestValue.setEnabled(false);
			integerUtilityHighestValue.setEnabled(false);
			if (!treeFrame.hasNoProfiles()) {
				integerMinField.setEnabled(false);
				integerMaxField.setEnabled(false);
			}
			integerUtilityLowestValue.setToolTipText("Disabled until there is a Utility Space.");
			integerUtilityHighestValue.setToolTipText("Disabled until there is a Utility Space.");
		} else {
			integerMinField.setEnabled(false);
			integerMaxField.setEnabled(false);
		}

		for (int i = 0; i < 4; i++) {
			panel.add(new JLabel());
		}

		return panel;
	}

	private JPanel constructRealPanel() {
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.PAGE_AXIS));
		JLabel label = new JLabel("Give the bounds of the Real values:");
		panel.add(label);

		JPanel min = new JPanel();
		min.add(new JLabel("Min: "));
		min.add(realMinField);
		panel.add(min);

		JPanel lin = new JPanel();
		lin.setAlignmentX(Component.LEFT_ALIGNMENT);
		lin.add(new JLabel("Linear: "));
		lin.add(realLinearField);
		panel.add(lin);

		JPanel par = new JPanel();
		par.setAlignmentX(Component.LEFT_ALIGNMENT);
		par.add(new JLabel("Constant: "));
		par.add(realParameterField);
		panel.add(par);

		JPanel max = new JPanel();
		max.add(new JLabel("Max: "));
		max.add(realMaxField);
		panel.add(max);

		if (((NegotiatorTreeTableModel) treeFrame.getTreeTable().getTree().getModel()).getUtilitySpace() == null) {
			realLinearField.setEnabled(false);
			realLinearField.setToolTipText("Disabled until there is a Utility Space.");
			realParameterField.setEnabled(false);
			realParameterField.setToolTipText("Disabled until there is a Utility Space.");
		}

		return panel;
	}

	protected boolean getWeightCheck() {
		return true;
	}

	/*
	 * 
	 * get values from the input thingy empty lines are not aloowed and just
	 * ignored..
	 */
	protected String[] getDiscreteValues() throws InvalidInputException {
		String[] values = discreteTextArea.getText().split("\n");
		return values;
	}

	/**
	 * Gets the evaluations for the discrete issue from the input field in this
	 * dialog.
	 * 
	 * @return An arrayList with the evaluations. Now returns elements with
	 *         value 0 to indicate non-entered (empty field) values.
	 *
	 * @throws InvalidInputException
	 *             if illegal input is given
	 */
	protected ArrayList<Integer> getDiscreteEvalutions() throws InvalidInputException, ClassCastException {
		String[] evalueStrings = discreteTextEvaluationArea.getText().split("\n", -1);

		ArrayList<Integer> evalues = new ArrayList<Integer>();
		for (int i = 0; i < evalueStrings.length; i++) {
			Integer value = 0;
			if (!evalueStrings[i].equals("")) {
				value = Integer.valueOf(evalueStrings[i]);
				if (value < 0)
					throw new InvalidInputException("Encountered " + value + ". Negative numbers are not allowed here");
			}
			evalues.add(value);
		}
		System.out.println(evalues);
		return evalues;
	}

	protected int getIntegerMin() throws InvalidInputException {
		if (!integerMinField.getText().equals(""))
			return Integer.parseInt(integerMinField.getText());
		else
			return 0;
	}

	protected int getIntegerOther() throws InvalidInputException {
		if (!integerOtherField.getText().equals(""))
			return Integer.parseInt(integerOtherField.getText());
		else
			return 0;
	}

	protected double getUtilityLowestInteger() throws InvalidInputException {
		if (!integerUtilityLowestValue.getText().equals(""))
			return Double.parseDouble(integerUtilityLowestValue.getText());
		else
			return 0;
	}

	protected double getUtilityHeighestInteger() throws InvalidInputException {
		if (!integerUtilityHighestValue.getText().equals(""))
			return Double.parseDouble(integerUtilityHighestValue.getText());
		else
			return 0;
	}

	protected int getIntegerMax() throws InvalidInputException {
		if (!integerMaxField.getText().equals(""))
			return Integer.parseInt(integerMaxField.getText());
		else
			return 0;
	}

	protected double getRealMin() throws InvalidInputException {
		if (!realMinField.getText().equals(""))
			return Double.parseDouble(realMinField.getText());
		else
			return 0.0;
	}

	protected double getRealOther() throws InvalidInputException {
		if (!realOtherField.getText().equals(""))
			return Double.parseDouble(realOtherField.getText());
		else
			return 0.0;
	}

	protected double getRealLinear() throws InvalidInputException {
		if (!realLinearField.getText().equals(""))
			return Double.parseDouble(realLinearField.getText());
		else
			return 0.0;
	}

	protected double getRealParameter() throws InvalidInputException {
		if (!realParameterField.getText().equals(""))
			return Double.parseDouble(realParameterField.getText());
		else
			return 0.0;
	}

	protected double getRealMax() throws InvalidInputException {
		if (!realMaxField.getText().equals(""))
			return Double.parseDouble(realMaxField.getText());
		else
			return 0.0;
	}

	protected Issue constructIssue() {
		return updateIssue(null);
	}

	/**
	 * This updates the data structures after the issue dialog was completed and
	 * user pressed OK. Not clear to me how it can return only an issue, so
	 * where are the values that were set as well? (the values should be put
	 * into a utility space)? The utility space is updated under water, and the
	 * dialog can access it via the parent node (treeFrame) that has access to
	 * the utility space....
	 * 
	 * @param issue
	 * @return the same issue as provided (but then updated).
	 * @throws exception
	 *             if issues can not be accepted. e.g. negative evaluation
	 *             values or if no evaluator available for issue while there is
	 *             a utiliyt space.
	 */
	protected Issue updateIssue(Issue issue) {
		// FIXME THIS CODE IS UGLY. The behaviour is not ok.
		String name;
		int number;
		String description;
		Objective selected = null; // The Objective that is selected in the
									// tree, which will be the new Issue's
									// parent.
		boolean newIssue = (issue == null); // Defines if a new Issue is added,
											// or if an existing Issue is being
											// edited.

		// Wouter: added: they threw away the old evaluator... bad because you
		// loose the weight settings of the evaluator.
		// Wouter; code is ugly. They createFrom a NEW evaluator anyway.
		// And at the end they check whethere there is a util space
		// anyway, and if not they throw away the new evaluator.....
		// Also we are paying here for the mix between domain and utility space
		// editor-in-one
		AdditiveUtilitySpace uts = treeFrame.getNegotiatorTreeTableModel().getUtilitySpace();
		Evaluator evaluator = null;
		if (uts != null && issue != null)
			evaluator = uts.getEvaluator(issue.getNumber());

		try {
			name = getObjectiveName();
			number = getObjectiveNumber();
			description = getObjectiveDescription();
		} catch (InvalidInputException e) {
			JOptionPane.showMessageDialog(this, e.getMessage());
			return null;
		}
		// If no issue is given to be modified,
		// construct a new one that is the child of the selected Objective.
		if (newIssue) {
			selected = treeFrame.getRoot();
		}

		String selectedType = (String) issueType.getSelectedItem();
		// Issue issue = null;
		if (selectedType == DISCRETE) {
			// EvaluatorDiscrete evDis = null;
			String[] values;
			ArrayList<Integer> evalues = null;
			try {
				values = getDiscreteValues();
			} catch (InvalidInputException e) {
				JOptionPane.showMessageDialog(this, e.getMessage());
				return null;
			}
			try {
				evalues = getDiscreteEvalutions();
				if (evalues == null)
					System.out.println("No evalues");
			} catch (Exception f) { // Can also be a casting exception.
				JOptionPane.showMessageDialog(this, "Problem reading evaluation values:" + f.getMessage());
			}

			if (newIssue) {
				issue = new IssueDiscrete(name, number, values);
			} else if (issue instanceof IssueDiscrete) {
				issue.setName(name);
				issue.setNumber(number);
				((IssueDiscrete) issue).clear();
				((IssueDiscrete) issue).addValues(values);
			}
			List<ValueDiscrete> v_enum = ((IssueDiscrete) issue).getValues();

			// load values into discrete evaluator
			if (evaluator != null && evalues != null) {
				try {
					((EvaluatorDiscrete) evaluator).clear();

					for (int i = 0; i < v_enum.size(); i++) {
						if (i < evalues.size()) // evalues field is 0 if error
												// occured at that field.
						{
							((EvaluatorDiscrete) evaluator).setEvaluation(((Value) v_enum.get(i)), evalues.get(i));
						}
					}
				} catch (Exception e) {
					JOptionPane.showMessageDialog(this, e.getMessage());
				}

				// Wouter: I don't like the way this works now but notime to
				// correct it.

				if (uts != null)
					uts.addEvaluator(issue, evaluator);
			}
		} else if (selectedType == INTEGER) {
			int min;
			int max;

			// Evaluator evInt = null;
			try {
				min = getIntegerMin();
				max = getIntegerMax();

				if (!integerUtilityLowestValue.getText().equals("")) {
					// evInt = new EvaluatorInteger();
					// evInt.setWeight(0.0);
					((EvaluatorInteger) evaluator).setLowerBound(min);
					((EvaluatorInteger) evaluator).setUpperBound(max);
					((EvaluatorInteger) evaluator).setLinearFunction(getUtilityLowestInteger(),
							getUtilityHeighestInteger());
				}
			} catch (InvalidInputException e) {
				JOptionPane.showMessageDialog(this, e.getMessage());
				return null;
			}
			if (newIssue) {
				issue = new IssueInteger(name, number, min, max);
			} else if (issue instanceof IssueInteger) {
				issue.setName(name);
				issue.setNumber(number);
				((IssueInteger) issue).setLowerBound(min);
				((IssueInteger) issue).setUpperBound(max);
			}
			if (uts != null)
				uts.addEvaluator(issue, evaluator);
		} else if (selectedType == REAL) {
			double min;
			double max;
			// Evaluator evReal = null;
			try {
				min = getRealMin();
				// other = getRealOther();
				max = getRealMax();
				if (!realLinearField.getText().equals("")) {
					// evReal = new EvaluatorReal();
					// evReal.setWeight(0.0);
					((EvaluatorReal) evaluator).setLowerBound(min);
					((EvaluatorReal) evaluator).setUpperBound(max);
					((EvaluatorReal) evaluator).setLinearParam(getRealLinear());
				} else if (!realParameterField.getText().equals("")) {
					// evReal = new EvaluatorReal();
					// evReal.setWeight(0.0);
					((EvaluatorReal) evaluator).setLowerBound(min);
					((EvaluatorReal) evaluator).setUpperBound(max);
					((EvaluatorReal) evaluator).setConstantParam(getRealParameter());
				}
			} catch (InvalidInputException e) {
				JOptionPane.showMessageDialog(this, e.getMessage());
				return null;
			}
			if (newIssue) {
				issue = new IssueReal(name, number, min, max);
			} else if (issue instanceof IssueReal) {
				issue.setName(name);
				issue.setNumber(number);
				((IssueReal) issue).setLowerBound(min);
				((IssueReal) issue).setUpperBound(max);
			}
			if (uts != null)
				uts.addEvaluator(issue, evaluator);

		} else {
			JOptionPane.showMessageDialog(this, "Please select an issue type!");
			return null;
		}

		issue.setDescription(description);
		if (newIssue) {
			selected.addChild(issue);
		}

		return issue;
	}

	/**
	 * Overrides actionPerformed from Objective.
	 */
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == okButton) {
			Issue issue = constructIssue();
			if (issue == null)
				return;
			else {
				// Notify the model that the contents of the treetable have
				// changed
				NegotiatorTreeTableModel model = (NegotiatorTreeTableModel) treeFrame.getNegotiatorTreeTableModel();
				Object[] path = { model.getRoot() };
				model.treeStructureChanged(this, path);
				this.dispose();
			}
		} else if (e.getSource() == cancelButton) {
			this.dispose();
		}
	}

	public void itemStateChanged(ItemEvent e) {
		((CardLayout) issuePropertyCards.getLayout()).show(issuePropertyCards, (String) e.getItem());
	}
}