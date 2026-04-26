package genius.gui;

import java.awt.Cursor;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URI;

import javax.swing.Icon;
import javax.swing.ImageIcon;

/**
 * The about menu created using Netbeans.
 * 
 * @author Mark Hendrikx
 */
public class About extends javax.swing.JFrame {

	private javax.swing.JLabel contributorsLabel;
	private javax.swing.JTextArea contributorsValueLabel;
	private javax.swing.JScrollPane contributorsValueScrollPane;
	private javax.swing.JTextArea geniusDescription;
	private javax.swing.JScrollPane geniusDescriptionScrollPane;
	private javax.swing.JLabel logo;
	private javax.swing.JLabel productVersionLabel;
	private javax.swing.JLabel productVersionValueLable;
	private javax.swing.JLabel websiteLabel;
	private javax.swing.JLabel websiteValueLabel;
	private final String GENIUSLINK = "http://ii.tudelft.nl/genius/";

	/**
	 * Creates new form About
	 */
	public About() {
		Toolkit tk = Toolkit.getDefaultToolkit();
		Dimension screenSize = tk.getScreenSize();
		this.setLocation(screenSize.width / 4, screenSize.height / 4);
		initComponents();
	}

	// <editor-fold defaultstate="collapsed" desc="Generated Code">
	private void initComponents() {

		contributorsValueScrollPane = new javax.swing.JScrollPane();
		contributorsValueLabel = new javax.swing.JTextArea();
		logo = new javax.swing.JLabel();
		productVersionLabel = new javax.swing.JLabel();
		websiteLabel = new javax.swing.JLabel();
		contributorsLabel = new javax.swing.JLabel();
		productVersionValueLable = new javax.swing.JLabel();
		websiteValueLabel = new javax.swing.JLabel();
		geniusDescriptionScrollPane = new javax.swing.JScrollPane();
		geniusDescription = new javax.swing.JTextArea();

		setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
		setResizable(false);

		contributorsValueScrollPane
				.setBackground(new java.awt.Color(255, 255, 255));

		contributorsValueLabel.setEditable(false);
		contributorsValueLabel.setColumns(20);
		contributorsValueLabel.setFont(new java.awt.Font("Calibri", 0, 14)); // NOI18N
		contributorsValueLabel.setLineWrap(true);
		contributorsValueLabel.setRows(5);
		contributorsValueLabel.setText(
				"R. Aydogan\nT. Baarslag\nA. Dirkzwager\nM. Hendrikx\nK. Hindriks\nW. Pasman\nD. Tykhonov\nD. Festen\nand others...");
		contributorsValueLabel.setWrapStyleWord(true);
		contributorsValueLabel.setOpaque(false);
		contributorsValueScrollPane.setViewportView(contributorsValueLabel);

		Icon icon = new ImageIcon(getClass().getClassLoader()
				.getResource("genius/gui/resources/GeniusLogo.png"));
		logo.setIcon(icon); // NOI18N

		productVersionLabel.setFont(new java.awt.Font("Calibri", 1, 18)); // NOI18N
		productVersionLabel.setText("Production version:");

		websiteLabel.setFont(new java.awt.Font("Calibri", 1, 18)); // NOI18N
		websiteLabel.setText("Website:");

		contributorsLabel.setFont(new java.awt.Font("Calibri", 1, 18)); // NOI18N
		contributorsLabel.setText("Contributors:");

		productVersionValueLable.setFont(new java.awt.Font("Calibri", 0, 18)); // NOI18N
		productVersionValueLable
				.setText(getClass().getPackage().getImplementationVersion());

		websiteValueLabel.setFont(new java.awt.Font("Calibri", 0, 18)); // NOI18N
		websiteValueLabel
				.setText("<html><a href=\"\">" + GENIUSLINK + "</a></html>");
		websiteValueLabel.setCursor(new Cursor(Cursor.HAND_CURSOR));
		websiteValueLabel.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				try {
					Desktop.getDesktop().browse(new URI(GENIUSLINK));
				} catch (Exception e2) {
				}
			}
		});

		geniusDescriptionScrollPane
				.setBackground(new java.awt.Color(255, 255, 255));
		geniusDescription.setFocusable(false);

		geniusDescription.setEditable(false);
		geniusDescription.setColumns(20);
		geniusDescription.setFont(new java.awt.Font("Calibri", 0, 20)); // NOI18N
		geniusDescription.setLineWrap(true);
		geniusDescription.setRows(5);
		geniusDescription.setText(
				"Genius is a negotiation environment that implements an open architecture for heterogeneous negotiating agents. Genius can be used to implement, or simulate, real life negotiations. This version includes a set of scenarios, negotiation strategies, and quality measures to quantify the performance of an agent.");
		geniusDescription.setWrapStyleWord(true);
		geniusDescription.setOpaque(false);
		geniusDescriptionScrollPane.setViewportView(geniusDescription);

		javax.swing.GroupLayout layout = new javax.swing.GroupLayout(
				getContentPane());
		getContentPane().setLayout(layout);
		layout.setHorizontalGroup(layout
				.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup().addContainerGap()
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.LEADING)
								.addGroup(
										javax.swing.GroupLayout.Alignment.TRAILING,
										layout.createSequentialGroup()
												.addGap(0, 0, Short.MAX_VALUE)
												.addComponent(logo,
														javax.swing.GroupLayout.PREFERRED_SIZE,
														582,
														javax.swing.GroupLayout.PREFERRED_SIZE))
								.addGroup(layout.createSequentialGroup()
										.addGroup(layout
												.createParallelGroup(
														javax.swing.GroupLayout.Alignment.LEADING)
												.addComponent(
														geniusDescriptionScrollPane)
												.addGroup(
														layout.createSequentialGroup()
																.addComponent(
																		contributorsLabel)
																.addPreferredGap(
																		javax.swing.LayoutStyle.ComponentPlacement.RELATED,
																		javax.swing.GroupLayout.DEFAULT_SIZE,
																		Short.MAX_VALUE)
																.addComponent(
																		contributorsValueScrollPane,
																		javax.swing.GroupLayout.PREFERRED_SIZE,
																		442,
																		javax.swing.GroupLayout.PREFERRED_SIZE))
												.addGroup(
														layout.createSequentialGroup()
																.addGroup(layout
																		.createParallelGroup(
																				javax.swing.GroupLayout.Alignment.LEADING)
																		.addGroup(
																				layout.createSequentialGroup()
																						.addComponent(
																								productVersionLabel)
																						.addPreferredGap(
																								javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
																						.addComponent(
																								productVersionValueLable,
																								javax.swing.GroupLayout.PREFERRED_SIZE,
																								43,
																								javax.swing.GroupLayout.PREFERRED_SIZE))
																		.addGroup(
																				layout.createSequentialGroup()
																						.addComponent(
																								websiteLabel)
																						.addPreferredGap(
																								javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
																						.addComponent(
																								websiteValueLabel,
																								javax.swing.GroupLayout.PREFERRED_SIZE,
																								390,
																								javax.swing.GroupLayout.PREFERRED_SIZE)))
																.addGap(0, 0,
																		Short.MAX_VALUE)))
										.addContainerGap()))));
		layout.setVerticalGroup(layout
				.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout
						.createSequentialGroup().addContainerGap()
						.addComponent(logo,
								javax.swing.GroupLayout.PREFERRED_SIZE, 234,
								javax.swing.GroupLayout.PREFERRED_SIZE)
						.addGap(13, 13, 13)
						.addComponent(geniusDescriptionScrollPane,
								javax.swing.GroupLayout.PREFERRED_SIZE,
								javax.swing.GroupLayout.DEFAULT_SIZE,
								javax.swing.GroupLayout.PREFERRED_SIZE)
						.addPreferredGap(
								javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(productVersionLabel)
								.addComponent(productVersionValueLable))
						.addGap(18, 18, 18)
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(websiteLabel)
								.addComponent(websiteValueLabel))
						.addGap(18, 18, 18)
						.addGroup(layout
								.createParallelGroup(
										javax.swing.GroupLayout.Alignment.LEADING)
								.addComponent(contributorsLabel)
								.addComponent(contributorsValueScrollPane,
										javax.swing.GroupLayout.PREFERRED_SIZE,
										javax.swing.GroupLayout.DEFAULT_SIZE,
										javax.swing.GroupLayout.PREFERRED_SIZE))
						.addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE,
								Short.MAX_VALUE)));
		setTitle("About");
		pack();
	}// </editor-fold>
}
