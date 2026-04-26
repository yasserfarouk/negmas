package genius.gui.boaframework;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JSeparator;
import javax.swing.JTextField;
import javax.swing.filechooser.FileFilter;

import genius.core.Global;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.repository.BOAagentRepository;
import genius.core.boaframework.repository.BOArepItem;
import genius.gui.panels.GenericFileFilter;

/**
 * GUI to add or edit a BOA component to the BOA repository.
 */
public class BOAComponentEditor extends JDialog {

	private static final long serialVersionUID = -7204112461104285605L;
	private JLabel componentNameLabel;
	private JTextField componentNameTextField;
	private JLabel componentClassLabel;
	private JTextField componentClassTextField;
	private JSeparator lowerSeparator;
	private JButton addComponent;
	private JButton openButton;
	private JSeparator upperSeparator;
	private BOArepItem result = null;
	private BoaType type;

	public BOAComponentEditor(Frame frame, String title) {
		super(frame, title, true);
		this.setLocation(frame.getLocation().x + frame.getWidth() / 2,
				frame.getLocation().y + frame.getHeight() / 4);
		this.setSize(frame.getSize().width / 3, frame.getSize().height / 2);
	}

	public BOArepItem getResult(BOArepItem item) {
		componentNameLabel = new JLabel("Component name");
		componentNameTextField = new JTextField();

		componentClassLabel = new JLabel("Component class");
		componentClassTextField = new javax.swing.JTextField();
		componentClassTextField.setEditable(false);

		addComponent = new JButton("Add component");
		addComponent.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				addComponent();
			}
		});

		openButton = new javax.swing.JButton("Open");
		openButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				openAction();
			}
		});

		upperSeparator = new javax.swing.JSeparator();
		lowerSeparator = new javax.swing.JSeparator();
		setResizable(false);

		javax.swing.GroupLayout layout = new javax.swing.GroupLayout(
				getContentPane());
		getContentPane().setLayout(layout);
		layout.setHorizontalGroup(layout
				.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addComponent(upperSeparator)
				.addGroup(layout.createSequentialGroup().addGap(12, 12, 12)
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.LEADING)
								.addGroup(layout.createSequentialGroup()
										.addComponent(componentNameLabel)
										.addGap(14, 14, 14)
										.addComponent(componentNameTextField))
								.addGroup(layout.createSequentialGroup()
										.addComponent(componentClassLabel)
										.addGap(18, 18, 18)
										.addComponent(componentClassTextField)
										.addPreferredGap(
												javax.swing.LayoutStyle.ComponentPlacement.RELATED)
										.addComponent(openButton))
								.addGroup(layout.createSequentialGroup()
										.addGroup(layout.createParallelGroup(
												javax.swing.GroupLayout.Alignment.LEADING,
												false)

												.addGroup(
														layout.createSequentialGroup()
																.addComponent(
																		addComponent)
																.addPreferredGap(
																		javax.swing.LayoutStyle.ComponentPlacement.RELATED)))
										.addGap(0, 0, Short.MAX_VALUE)))
						.addContainerGap())
				.addComponent(lowerSeparator));
		layout.setVerticalGroup(layout
				.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup().addContainerGap()
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(componentNameLabel)
								.addComponent(componentNameTextField,
										javax.swing.GroupLayout.PREFERRED_SIZE,
										javax.swing.GroupLayout.DEFAULT_SIZE,
										javax.swing.GroupLayout.PREFERRED_SIZE))
						.addPreferredGap(
								javax.swing.LayoutStyle.ComponentPlacement.RELATED)

						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(componentClassLabel)
								.addComponent(componentClassTextField,
										javax.swing.GroupLayout.PREFERRED_SIZE,
										javax.swing.GroupLayout.DEFAULT_SIZE,
										javax.swing.GroupLayout.PREFERRED_SIZE)
								.addComponent(openButton))
						.addPreferredGap(
								javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addComponent(upperSeparator,
								javax.swing.GroupLayout.PREFERRED_SIZE, 10,
								javax.swing.GroupLayout.PREFERRED_SIZE)
						.addPreferredGap(
								javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addComponent(lowerSeparator,
								javax.swing.GroupLayout.PREFERRED_SIZE, 10,
								javax.swing.GroupLayout.PREFERRED_SIZE)
						.addPreferredGap(
								javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.LEADING)
								.addComponent(addComponent))
						.addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE,
								Short.MAX_VALUE)));

		if (item != null) {
			componentNameTextField.setText(item.getName());
			componentClassTextField.setText(item.getClassPath());
			type = item.getType();
			// for (BOAparameter param : item.getParameters()) {
			// parameterListModel.addElement(param);
			// }
			addComponent.setText("Save");
		}

		pack();
		setVisible(true);
		return result;
	}

	private void addComponent() {
		boolean valid = true;
		if (componentNameTextField.getText().length() == 0
				|| componentNameTextField.getText().length() > 35) {
			valid = false;
			JOptionPane.showMessageDialog(null,
					"Component name should be non-empty and at most 35 characters.",
					"Invalid parameter input", 0);
		}
		if (componentClassTextField.getText().length() == 0) {
			valid = false;
			JOptionPane.showMessageDialog(null, "Please specify a class.",
					"Invalid parameter input", 0);
		}
		if (valid) {
			String name = componentNameTextField.getText();
			String classPath = componentClassTextField.getText();
			BOArepItem newComponent = new BOArepItem(name, classPath, type);
			// for (int i = 0; i < parameterListModel.getSize(); i++) {
			// BOAparameter item = (BOAparameter) parameterListModel
			// .getElementAt(i);
			// newComponent.addParameter(item);
			// }
			BOAagentRepository.getInstance().addComponent(newComponent);
			result = newComponent;
			dispose();
		}
	}

	private void openAction() {

		JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));

		// Filter such that only directories and .class files are shown.
		FileFilter filter = new GenericFileFilter("class",
				"Java class files (.class)");
		fc.setFileFilter(filter);

		// Open the file picker
		int returnVal = fc.showOpenDialog(null);

		// If file selected
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			File file = fc.getSelectedFile();

			try {
				Object object = Global.loadClassFromFile(file);
				if (object instanceof OfferingStrategy) {
					type = BoaType.BIDDINGSTRATEGY;
				} else if (object instanceof AcceptanceStrategy) {
					type = BoaType.ACCEPTANCESTRATEGY;
				} else if (object instanceof OpponentModel) {
					type = BoaType.OPPONENTMODEL;
				} else if (object instanceof OMStrategy) {
					type = BoaType.OMSTRATEGY;
				} else {
					throw new IllegalArgumentException("File " + file
							+ " does not extend OfferingStrategy, AcceptanceStrategy, \n"
							+ "OpponentModel, or OMStrategy.");
				}
			} catch (Throwable e) {
				Global.showLoadError(file, e);
				return;
			}

			try {
				componentClassTextField.setText(file.getCanonicalPath());
				if (componentNameTextField.getText().isEmpty()) {
					componentNameTextField.setText(file.getName());
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}
}