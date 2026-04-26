package genius.gui.domainrepository;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;

/**
 * GUI used to createFrom a new domain.
 *
 * @author Mark Hendrikx
 */
public class CreateNewDomain extends JDialog {

	private JButton okButton;
	private JButton cancelButton;
	private JLabel domainNameLabel;
	private JTextField domainNameField;
	private String result = null;

	/**
	 * Creates new form DomainCreationUI
	 */
	public CreateNewDomain(Frame frame) {
		super(frame, "Create domain", true);
		this.setLocation(frame.getLocation().x + frame.getWidth() / 4,
				frame.getLocation().y + frame.getHeight() / 4);
	}

	public String getResult() {
		setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
		setResizable(false);

		domainNameLabel = new JLabel();
		okButton = new JButton();
		cancelButton = new JButton();
		domainNameField = new JTextField();

		domainNameLabel.setText("Domain name");

		okButton.setText("Ok");
		okButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (domainNameField.getText().equals("")) {
					JOptionPane.showMessageDialog(null,
							"The domain name may not be empty.",
							"Parameter error", 0);
				} else {
					result = domainNameField.getText();
					dispose();
				}
			}
		});

		cancelButton.setText("Cancel");
		cancelButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				dispose();
			}
		});

		javax.swing.GroupLayout layout = new javax.swing.GroupLayout(
				getContentPane());
		getContentPane().setLayout(layout);
		layout.setHorizontalGroup(layout
				.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup().addContainerGap()
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.LEADING)
								.addGroup(layout.createSequentialGroup()
										.addComponent(domainNameLabel)
										.addGap(74, 74, 74)
										.addComponent(domainNameField))
								.addGroup(layout.createSequentialGroup()
										.addComponent(okButton,
												javax.swing.GroupLayout.PREFERRED_SIZE,
												80,
												javax.swing.GroupLayout.PREFERRED_SIZE)
										.addPreferredGap(
												javax.swing.LayoutStyle.ComponentPlacement.RELATED)
										.addComponent(cancelButton,
												javax.swing.GroupLayout.PREFERRED_SIZE,
												80,
												javax.swing.GroupLayout.PREFERRED_SIZE)
										.addGap(0, 133, Short.MAX_VALUE)))
						.addContainerGap()));
		layout.setVerticalGroup(layout
				.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup().addContainerGap()
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(domainNameLabel)
								.addComponent(domainNameField,
										javax.swing.GroupLayout.PREFERRED_SIZE,
										javax.swing.GroupLayout.DEFAULT_SIZE,
										javax.swing.GroupLayout.PREFERRED_SIZE))
						.addPreferredGap(
								javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(okButton)
								.addComponent(cancelButton))
						.addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE,
								Short.MAX_VALUE)));

		pack();
		setVisible(true);
		return result;
	}
}