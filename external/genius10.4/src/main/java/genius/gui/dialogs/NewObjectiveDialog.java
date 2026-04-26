package genius.gui.dialogs;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import genius.core.issue.Objective;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorObjective;
import genius.gui.tree.NegotiatorTreeTableModel;
import genius.gui.tree.DomainAndProfileEditorPanel;

/**
 * A dialog allowing the user to add a new Objective
 */

public class NewObjectiveDialog extends JDialog implements ActionListener {

	private static final long serialVersionUID = 4665460273597895313L;

	protected JButton okButton;
	protected JButton cancelButton;
	protected JLabel nameLabel;
	protected JLabel numberLabel;
	protected JTextField nameField;
	protected JTextField numberField;
	protected DomainAndProfileEditorPanel treeFrame;

	public NewObjectiveDialog(DomainAndProfileEditorPanel owner) {
		this(owner, false);
	}

	/**
	 * 
	 * @param owner
	 * @param modal
	 *            true if multiple dialogs can be open at once, false if not.
	 */
	public NewObjectiveDialog(DomainAndProfileEditorPanel owner, boolean modal) {
		this(owner, modal, "Create new Objective");
	}

	public NewObjectiveDialog(DomainAndProfileEditorPanel owner, boolean modal, String name) {
		super();
		this.treeFrame = owner;

		initPanels();

		this.pack();
		this.setVisible(true);
	}

	protected void initPanels() {
		this.setLayout(new BorderLayout());

		this.add(constructBasicPropertyPanel(), BorderLayout.NORTH);
		this.add(constructButtonPanel(), BorderLayout.SOUTH);
	}

	private JPanel constructBasicPropertyPanel() {
		// Initialize the labels
		nameLabel = new JLabel("Name:");
		nameLabel.setAlignmentX(Component.RIGHT_ALIGNMENT);
		numberLabel = new JLabel("Number:");
		numberLabel.setAlignmentX(Component.RIGHT_ALIGNMENT);

		// Initialize the fields
		nameField = new JTextField();
		nameField.setAlignmentX(Component.LEFT_ALIGNMENT);
		numberField = new JTextField();
		numberField.setAlignmentX(Component.LEFT_ALIGNMENT);

		JPanel labelPanel = new JPanel();
		labelPanel.setLayout(new BoxLayout(labelPanel, BoxLayout.PAGE_AXIS));

		labelPanel.add(new JLabel("Name:"));

		JPanel fieldPanel = new JPanel();
		fieldPanel.setLayout(new BoxLayout(fieldPanel, BoxLayout.PAGE_AXIS));

		fieldPanel.add(nameField);

		JPanel basicPropertyPanel = new JPanel();
		basicPropertyPanel.setBorder(
				BorderFactory.createTitledBorder("Basic Properties"));
		basicPropertyPanel.setLayout(new BorderLayout());
		basicPropertyPanel.add(labelPanel, BorderLayout.LINE_START);
		basicPropertyPanel.add(fieldPanel, BorderLayout.CENTER);

		return basicPropertyPanel;
	}

	/**
	 * Initializes the buttons, and returns a panel containing them.
	 * 
	 * @return a JPanel with the buttons.
	 */
	private JPanel constructButtonPanel() {
		// Initialize the buttons
		okButton = new JButton("Ok");
		okButton.addActionListener(this);
		cancelButton = new JButton("Cancel");
		cancelButton.addActionListener(this);

		JPanel buttonPanel = new JPanel();
		buttonPanel.add(okButton);
		buttonPanel.add(cancelButton);

		return buttonPanel;
	}

	protected String getObjectiveName() throws InvalidInputException {
		return nameField.getText();
	}

	protected int getObjectiveNumber() throws InvalidInputException {
		return (((NegotiatorTreeTableModel) treeFrame.getTreeTable().getTree()
				.getModel()).getHighestObjectiveNr() + 1);
	}

	protected String getObjectiveDescription() throws InvalidInputException {
		return "";
	}

	// overridden by issues
	protected boolean getWeightCheck() {
		return false;
	}

	protected Objective constructObjective() {
		String name;
		int number;
		Objective selected; // The Objective that is seleced in the tree, which
							// will be the new Objective's parent.
		try {
			name = getObjectiveName();
			number = treeFrame.getNegotiatorTreeTableModel()
					.getHighestObjectiveNr() + 1;
		} catch (InvalidInputException e) {
			JOptionPane.showMessageDialog(this, e.getMessage());
			return null;
		}
		try {
			selected = (Objective) treeFrame.getTreeTable().getTree()
					.getLastSelectedPathComponent();
			if (selected == null) {
				JOptionPane.showMessageDialog(this,
						"There is no valid parent selected for this objective.");
				return null;
			}
		} catch (Exception e) {
			JOptionPane.showMessageDialog(this,
					"There is no valid parent selected for this objective.");
			return null;
		}
		Objective objective = new Objective(selected, name, number);

		selected.addChild(objective);
		AdditiveUtilitySpace utilspace = treeFrame.getNegotiatorTreeTableModel()
				.getUtilitySpace();
		if (utilspace != null) {
			EvaluatorObjective ev = new EvaluatorObjective();
			ev.setHasWeight(getWeightCheck());
			utilspace.addEvaluator(objective, ev);
		}

		return objective;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == okButton) {
			Objective objective = constructObjective();
			if (objective == null)
				return;

			NegotiatorTreeTableModel model = treeFrame
					.getNegotiatorTreeTableModel();
			model.treeStructureChanged(this, treeFrame.getTreeTable().getTree()
					.getSelectionPath().getPath());

			this.dispose();

		} else if (e.getSource() == cancelButton) {
			this.dispose();
		}
	}

	protected class InvalidInputException extends Exception {

		private static final long serialVersionUID = 866805428763450366L;

		protected InvalidInputException() {
			super();
		}

		protected InvalidInputException(String message) {
			super(message);
		}
	}
}