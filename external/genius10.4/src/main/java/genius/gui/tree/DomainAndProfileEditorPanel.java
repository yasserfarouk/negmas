package genius.gui.tree;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;

import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.TableColumnModel;

import genius.core.DomainImpl;
import genius.core.issue.Issue;
import genius.core.issue.Objective;
import genius.core.jtreetable.JTreeTable;
import genius.core.repository.DomainRepItem;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UncertainAdditiveUtilitySpace;
import genius.gui.dialogs.EditIssueDialog;
import genius.gui.dialogs.NewIssueDialog;

/**
 * Panel to edit a domain and a profile. It configures itself to domain or
 * profile editor, depending on whether there are already profiles available for
 * the domain.
 */
public class DomainAndProfileEditorPanel extends JPanel {

	private static final long serialVersionUID = 9072786889017106286L;
	// Attributes
	private static final Color UNSELECTED = Color.WHITE;
	private static final Color HIGHLIGHT = Color.YELLOW;
	private JTreeTable treeTable;
	private NegotiatorTreeTableModel model;
	private JMenuBar menuBar;
	private JMenu fileMenu;
	private JMenu editMenu;
	private DomainRepItem fDomainRepItem;
	/** true iff no profiles for domain yet and the domain can be edited. */
	private boolean hasNoProfiles;
	private JTextField discount;
	private JTextField reservationValue;

	private UncertaintySettingsModel uncertaintySettings;

	/**
	 * Create new profile
	 * 
	 * @param domain
	 * @param hasNoProfiles
	 *            if true, then there are no profiles for the domain yet and you
	 *            can edit the domain.
	 */
	public DomainAndProfileEditorPanel(DomainImpl domain,
			boolean hasNoProfiles) {
		this(new NegotiatorTreeTableModel(domain), hasNoProfiles);
	}

	/**
	 * Edit existing profile
	 * 
	 * @param domain
	 *            the domain of the profile
	 * @param utilitySpace
	 *            the profile to edit
	 */
	public DomainAndProfileEditorPanel(DomainImpl domain,
			AdditiveUtilitySpace utilitySpace) {
		this(new NegotiatorTreeTableModel(domain, utilitySpace), false);
	}

	public DomainAndProfileEditorPanel(NegotiatorTreeTableModel treeModel,
			boolean hasNoProfiles) {
		super();
		this.hasNoProfiles = hasNoProfiles;
		init(treeModel, null);
	}

	public void clearTreeTable(DomainImpl domain,
			AdditiveUtilitySpace utilitySpace) {
		init(new NegotiatorTreeTableModel(domain, utilitySpace),
				this.getSize());
	}

	public boolean isDomain() {
		return model.getUtilitySpace() == null;
	}

	public boolean hasNoProfiles() {
		return hasNoProfiles;
	}

	public JTreeTable getTreeTable() {
		return treeTable;
	}

	public NegotiatorTreeTableModel getNegotiatorTreeTableModel() {
		return model;
	}

	public Objective getRoot() {
		return (Objective) model.getRoot();
	}

	/*******************************************************/
	/*********** PRIVATE CODE ******************************/
	/*******************************************************/

	private void init(NegotiatorTreeTableModel treeModel, Dimension size) {
		final DomainAndProfileEditorPanel thisFrame = this;
		model = treeModel;
		setLayout(new BorderLayout());

		initTable(model);
		treeTable.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseReleased(MouseEvent e) {

				if (e.getClickCount() == 2) {
					Object selected = treeTable.getTree()
							.getLastSelectedPathComponent();

					if (selected instanceof Issue) {
						if (hasNoProfiles || !isDomain()) {
							new EditIssueDialog(thisFrame, (Issue) selected);
						} else {
							showError(
									"You may only edit the issues when there are no preference profiles.");
						}
					}
				}
			}
		});
		treeTable.setRowHeight(40);
		// Initialize the Menu
		initMenus();
		JPanel bottomArea = new JPanel(new BorderLayout());

		AdditiveUtilitySpace utilspace = model.getUtilitySpace();
		if (utilspace != null) {
			uncertaintySettings = new UncertaintySettingsModel(utilspace);
			bottomArea.add(new UncertaintySettingsPanel(uncertaintySettings),
					BorderLayout.CENTER);
		}
		bottomArea.add(savePanel(), BorderLayout.SOUTH);

		add(bottomArea, BorderLayout.SOUTH);

		if (size != null)
			this.setSize(size);

	}

	private void showError(String error) {
		JOptionPane.showMessageDialog(null, error, "Edit error", 0);

	}

	private JPanel savePanel() {
		JButton saveButton = new JButton("Save changes");
		Icon icon = new ImageIcon(getClass().getClassLoader()
				.getResource("genius/gui/resources/save.png"));
		saveButton.setPreferredSize(new Dimension(180, 60));
		saveButton.setIcon(icon);
		saveButton.setFont(saveButton.getFont().deriveFont(14.0f));
		saveButton.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				save();
			}

		});

		JPanel simplePanel = new JPanel();
		if (hasNoProfiles) {
			addDomainEditButtons(simplePanel);
		}
		simplePanel.add(saveButton);

		if (model.getUtilitySpace() != null) {
			addProfileEditButtons(simplePanel);
		}
		return simplePanel;
	}

	private void addProfileEditButtons(JPanel simplePanel) {
		simplePanel.add(new JLabel("Discount: "));
		discount = new JTextField(
				model.getUtilitySpace().getDiscountFactor() + "", 5);
		simplePanel.add(discount);

		simplePanel.add(new JLabel("Reservation value: "));
		reservationValue = new JTextField(
				model.getUtilitySpace().getReservationValueUndiscounted() + "",
				5);
		simplePanel.add(reservationValue);
	}

	private void addDomainEditButtons(JPanel simplePanel) {
		final DomainAndProfileEditorPanel thisFrame = this;
		JButton addIssue = new JButton("Add issue");
		Icon icon2 = new ImageIcon(getClass().getClassLoader()
				.getResource("genius/gui/resources/edit_add-32.png"));
		addIssue.setPreferredSize(new Dimension(180, 60));
		addIssue.setIcon(icon2);
		addIssue.setFont(addIssue.getFont().deriveFont(18.0f));
		addIssue.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				new NewIssueDialog(thisFrame);
			}
		});
		simplePanel.add(addIssue);

		JButton removeIssue = new JButton("Remove issue");
		Icon icon3 = new ImageIcon(getClass().getClassLoader()
				.getResource("genius/gui/resources/edit_remove-32.png"));
		removeIssue.setPreferredSize(new Dimension(180, 60));
		removeIssue.setIcon(icon3);
		removeIssue.setFont(removeIssue.getFont().deriveFont(18.0f));
		removeIssue.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				Object selected = treeTable.getTree()
						.getLastSelectedPathComponent();

				if (selected instanceof Issue) {
					((Issue) selected).removeFromParent();
					// correct numbering
					for (int i = 0; i < model.getDomain().getIssues()
							.size(); i++) {
						model.getDomain().getIssues().get(i).setNumber(i + 1); // +
																				// 1
																				// for
																				// root
					}
					treeTable.updateUI();
				}
			}
		});
		simplePanel.add(removeIssue);
	}

	private void save() {
		if (model.getUtilitySpace() != null) {
			saveProfile();
		} else {
			model.getDomain().toXML().saveToFile(model.getDomain().getName());
		}
	}

	private void saveProfile() {
		double newDiscount = 1.0;
		AdditiveUtilitySpace uspace;
		// create the right new class, depending on uncertainty setting
		if (uncertaintySettings.getIsEnabled().getValue()) {
			int comps = uncertaintySettings.getComparisons().getValue();
			int errs = uncertaintySettings.getErrors().getValue();
			double eli = uncertaintySettings.getElicitationCost().getValue();
			boolean fixedseed = uncertaintySettings.getIsFixedSeed().getValue();
			uspace = new UncertainAdditiveUtilitySpace(model.getUtilitySpace(),
					comps, errs, eli, fixedseed,
					uncertaintySettings.getIsExperimental().getValue());
		} else {
			uspace = new AdditiveUtilitySpace(model.getUtilitySpace());
		}

		// FIXME GUI should not allow to set illegal values in first place.
		try {
			newDiscount = Double.parseDouble(discount.getText());
			if (newDiscount < 0 || newDiscount > 1) {
				showError("The discount value is not valid.");
				return;
			}
		} catch (Exception e) {
			showError("The discount value is not valid.");
			return;
		}
		double newRV = 1.0;
		try {
			newRV = Double.parseDouble(reservationValue.getText());
			if (newRV < 0 || newRV > 1) {
				showError("The reservation value is not valid.");
				return;
			}
		} catch (Exception e) {
			showError("The reservation value is not valid.");
			return;
		}
		uspace.setDiscount(newDiscount);
		uspace.setReservationValue(newRV);
		try {
			uspace.toXML().saveToFile(model.getUtilitySpace().getFileName());
		} catch (IOException e) {
			showError("Something went wrong while saving:" + e.getMessage());
			e.printStackTrace();
		}
	}

	private void initTable(NegotiatorTreeTableModel model) {
		treeTable = new JTreeTable(model);
		treeTable.setPreferredSize(new Dimension(1024, 800));
		treeTable.setPreferredScrollableViewportSize(new Dimension(1024, 300));
		treeTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		treeTable.setRowSelectionAllowed(true);
		treeTable.setColumnSelectionAllowed(false);
		treeTable.setCellSelectionEnabled(true);

		TableColumnModel colModel = treeTable.getColumnModel();
		if (treeTable.getColumnCount() > 3)
			colModel.getColumn(3).setMinWidth(220); // Wouter: make it likely
													// that Weight column is
													// shown completely.

		DefaultTableCellRenderer labelRenderer = new JLabelCellRenderer();
		treeTable.setDefaultRenderer(JLabel.class, labelRenderer);
		treeTable.setDefaultRenderer(JTextField.class, labelRenderer);

		IssueValueCellEditor valueEditor = new IssueValueCellEditor(model);
		treeTable.setDefaultRenderer(IssueValuePanel.class, valueEditor);
		treeTable.setDefaultEditor(IssueValuePanel.class, valueEditor);

		WeightSliderCellEditor cellEditor = new WeightSliderCellEditor(model);
		treeTable.setDefaultRenderer(WeightSlider.class, cellEditor);
		treeTable.setDefaultEditor(WeightSlider.class, cellEditor);
		treeTable.setRowHeight(24);

		JScrollPane treePane = new JScrollPane(treeTable);
		treePane.setBackground(treeTable.getBackground());
		add(treePane, BorderLayout.CENTER);
	}

	private void initMenus() {
		menuBar = new JMenuBar();
		fileMenu = new JMenu("File");
		editMenu = new JMenu("Edit");
		menuBar.add(fileMenu);
		menuBar.add(editMenu);
	}

}