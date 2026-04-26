package negotiator.parties;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Frame;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.WindowConstants;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.exceptions.Warning;
import genius.core.parties.NegotiationParty;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Ask user to enter bid details. Modal dialog. only works with
 * {@link AdditiveUtilitySpace}
 * 
 * @author W.Pasman
 */
public class EnterBidDialog2 extends JDialog
{
	// we have some whitespace in the buttons,
	// that makes nicer buttons and also artificially increases the window size.
	public static final String DO_BID = "       Do Bid       ";
	public static final String END_NEGOTIATION = "End Negotiation";
	public static final String ACCEPT = " Accept Opponent Bid ";

	private static final long serialVersionUID = -8582527630534972700L;
	private NegoInfo negoinfo; // the table model
	private genius.core.actions.Action selectedAction;
	private NegotiationParty party;
	private JTextArea negotiationMessages = new JTextArea("NO MESSAGES YET");
	private JButton buttonAccept = button(ACCEPT);
	private JButton buttonEnd = button(END_NEGOTIATION);
	private JButton buttonBid = button(DO_BID);
	private JPanel buttonPanel = new JPanel();
	private JTable BidTable;
	private Bid lastOppBid;
	private AgentID partyID;

	/**
	 * Create dialog but do not yet show it. Call {@link #setVisible(boolean)}
	 * to show.
	 * 
	 * @param party
	 * @param parent
	 * @param modal
	 * @param us
	 * @param lastOppBid
	 *            last oppponent bid that can be accepted, or null if no such
	 *            bid.
	 * @throws Exception
	 */
	public EnterBidDialog2(NegotiationParty party, AgentID id, Frame parent, boolean modal, AdditiveUtilitySpace us,
			Bid lastOppBid) throws Exception {
		super(parent, modal);
		this.party = party;
		this.partyID = id;
		this.lastOppBid = lastOppBid;
		negoinfo = new NegoInfo(null, null, us);
		initThePanel();
	}

	private JButton button(String name) {
		JButton button = new JButton(name);
		button.setName(name);
		return button;
	}

	private void initThePanel() {
		if (negoinfo == null)
			throw new NullPointerException("negoinfo is null");
		Container pane = getContentPane();
		pane.setLayout(new BorderLayout());
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		setTitle("Choose action for party " + partyID.toString());

		// createFrom north field: the message field
		pane.add(negotiationMessages, "North");

		// createFrom center panel: the bid table
		BidTable = new JTable(negoinfo);
		BidTable.setGridColor(Color.lightGray);
		JPanel tablepane = new JPanel(new BorderLayout());
		tablepane.add(BidTable.getTableHeader(), "North");
		tablepane.add(BidTable, "Center");
		pane.add(tablepane, "Center");
		BidTable.setRowHeight(35);

		// createFrom south panel: the buttons:
		buttonPanel.setLayout(new FlowLayout());
		buttonPanel.add(buttonEnd);
		buttonAccept.setEnabled(lastOppBid != null);
		buttonPanel.add(buttonAccept);
		buttonPanel.add(buttonBid);
		pane.add(buttonPanel, "South");
		buttonBid.setSelected(true);

		// set action listeners for the buttons
		buttonBid.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				buttonBidActionPerformed(evt);
			}
		});
		buttonEnd.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				buttonEndActionPerformed(evt);
			}
		});
		buttonAccept.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				buttonAcceptActionPerformed(evt);
			}
		});
		pack(); // pack will do complete layout, getting all cells etc.
	}

	private Bid getBid() {
		Bid bid = null;
		try {
			bid = negoinfo.getBid();
		} catch (Exception e) {
			JOptionPane.showMessageDialog(null, "There is a problem with your bid: " + e.getMessage());
		}
		return bid;
	}

	private void buttonBidActionPerformed(java.awt.event.ActionEvent evt) {

		Bid bid = getBid();
		if (bid != null) {
			selectedAction = new Offer(partyID, bid);
			setVisible(false);
		}
	}

	private void buttonAcceptActionPerformed(java.awt.event.ActionEvent evt) {
		Bid bid = getBid();
		if (bid != null) {
			System.out.println("Accept performed");
			selectedAction = new Accept(partyID, lastOppBid);
			setVisible(false);
		}
	}

	private void buttonEndActionPerformed(java.awt.event.ActionEvent evt) {
		System.out.println("End Negotiation performed");
		selectedAction = new EndNegotiation(partyID);
		setVisible(false);
	}

	public genius.core.actions.Action askUserForAction(genius.core.actions.Action opponentAction, Bid myPreviousBid,
			Bid mostRecentOffer) {
		negoinfo.lastAccepted = mostRecentOffer;
		return askUserForAction(opponentAction, myPreviousBid);
	}

	/**
	 * This is called by UIAgent repeatedly, to ask for next action.
	 * 
	 * @param opponentAction
	 *            is action done by opponent
	 * @param myPreviousBid
	 * @return our next negotionat action.
	 */
	public genius.core.actions.Action askUserForAction(genius.core.actions.Action opponentAction, Bid myPreviousBid) {

		setTitle("Choose action for party " + partyID.toString());
		if (opponentAction == null) {
			negotiationMessages.setText("Opponent did not send any action.");
		}
		if (opponentAction instanceof Accept) {
			negotiationMessages.setText("Opponent accepted your last bid!");
		}
		if (opponentAction instanceof EndNegotiation) {
			negotiationMessages.setText("Opponent cancels the negotiation.");
		}
		if (opponentAction instanceof Offer) {
			negotiationMessages.setText("Opponent proposes the following bid:");
		}
		try {
			negoinfo.setOurBid(myPreviousBid);
		} catch (Exception e) {
			new Warning("error in askUserForAction:", e, true, 2);
		}

		BidTable.setDefaultRenderer(BidTable.getColumnClass(0), new MyCellRenderer1(negoinfo));
		BidTable.setDefaultEditor(BidTable.getColumnClass(0), new MyCellEditor(negoinfo));

		pack();
		setVisible(true); // this returns only after the panel closes.
		// Wouter: this WILL return normally if Thread is killed, and the
		// ThreadDeath exception will disappear.
		return selectedAction;
	}
}

/********************************************************/
