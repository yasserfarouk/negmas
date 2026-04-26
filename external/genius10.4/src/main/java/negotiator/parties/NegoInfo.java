package negotiator.parties;

import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JProgressBar;
import javax.swing.JTextArea;
import javax.swing.table.AbstractTableModel;

import genius.core.Bid;
import genius.core.exceptions.Warning;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

/**
 * NegoInfo is the class that contains all the negotiation data, and handles the
 * GUI, updating the JTable. This is the main interface to the actual JTable.
 * This is usually called XXXModel but I dont like the 'model' in the name. We
 * implement actionlistener to hear the combo box events that require
 * re-rendering of the total cost and utility field. We are pretty hard-wired
 * for a 3-column table, with column 0 the labels, column 1 the opponent bid and
 * col2 our own bid.
 * 
 * <p>
 * Many dialogs in negotiator.parties package use this.
 */
@SuppressWarnings("serial")
class NegoInfo extends AbstractTableModel implements ActionListener {
	public Bid ourOldBid; // Bid is hashmap <issueID,Value>. Our current bid is
							// only in the comboboxes,
							// use getBid().
	public Bid lastAccepted;
	public AdditiveUtilitySpace utilitySpace; // WARNING: this may be null
	public List<Issue> issues = new ArrayList<Issue>();
	// the issues, in row order as in the GUI. Init to empty, to enable
	// freshly initialized NegoInfo to give useful results to the GUI.
	public ArrayList<Integer> IDs; // the IDs/numbers of the issues, ordered to
									// row number
	public ArrayList<JComboBox> comboBoxes; // the combo boxes for the second
											// column, ordered to row number

	NegoInfo(Bid our, Bid lastAccepted, AdditiveUtilitySpace us)
			throws Exception {
		// Wouter: just discovered that assert does nothing...........
		// David@Wouter: Assert only works when assert compile flag is set to
		// true
		ourOldBid = our;
		this.lastAccepted = lastAccepted;
		utilitySpace = us;
		issues = utilitySpace.getDomain().getIssues();
		IDs = new ArrayList<Integer>();
		for (int i = 0; i < issues.size(); i++)
			IDs.add(issues.get(i).getNumber());
		makeComboBoxes();
	}

	@Override
	public int getColumnCount() {
		return 3;
	}

	@Override
	public int getRowCount() {
		// the extra row shows the utility of the bids.
		return issues.size() + 1;
	}

	@Override
	public boolean isCellEditable(int row, int col) {
		return (col == 2 && row < issues.size());
	}

	private String[] colNames = { "Issue", "Last Accepted Bid", "Your bid" };

	@Override
	public String getColumnName(int col) {
		return colNames[col];
	}

	public void setOurBid(Bid bid) throws Exception {
		ourOldBid = bid;
		if (bid == null)
			try {
				ourOldBid = utilitySpace.getMaxUtilityBid();
			} catch (Exception e) {
				System.out.println("error getting max utility first bid:"
						+ e.getMessage());
				e.printStackTrace();
			}
		makeComboBoxes(); // reset all
		setComboBoxes(ourOldBid);
	}

	void makeComboBoxes() throws Exception {
		comboBoxes = new ArrayList<JComboBox>();
		for (Issue issue : issues) {
			if (!(issue instanceof IssueDiscrete))
				throw new IllegalArgumentException("Not supported issue "
						+ issue + ": not IssueDiscrete. ");
			List<ValueDiscrete> values = ((IssueDiscrete) issue).getValues();
			JComboBox cbox = new JComboBox();
			EvaluatorDiscrete eval = null;
			if (utilitySpace != null)
				eval = (EvaluatorDiscrete) utilitySpace.getEvaluator(issue
						.getNumber());
			for (ValueDiscrete val : values) {
				String utilinfo = "";
				if (eval != null)
					try {
						// utilinfo="("+eval.getEvaluation(val)+")";
						utilinfo = "(" + eval.getValue(val) + ")";

					} catch (Exception e) {
						System.out.println("no evaluator for " + val + "???");
					}

				cbox.addItem(val + utilinfo);
			}
			comboBoxes.add(cbox);
			for (JComboBox b : comboBoxes)
				b.addActionListener(this);
		}
	}

	/**
	 * set the initial combo box selections according to ourOldBid Note, we can
	 * only handle Discrete evaluators right now.
	 * 
	 * @throws if
	 *             there is a problem with the issues and evaluators.
	 */
	void setComboBoxes(Bid bid) throws Exception {
		for (int i = 0; i < issues.size(); i++) {
			IssueDiscrete iss = (IssueDiscrete) issues.get(i);
			ValueDiscrete val = (ValueDiscrete) bid.getValue(iss.getNumber());
			comboBoxes.get(i).setSelectedIndex(
					((IssueDiscrete) iss).getValueIndex(val));
		}
	}

	/**
	 * get the currently chosen evaluation value of given row in the table.
	 * 
	 * @param bid
	 *            : which bid (the column in the table are for ourBid and
	 *            opponentBid)
	 * @return the evaluation of the given row in the bid. returns null if the
	 *         bid has no value in that row.
	 * @throws probablly
	 *             if rownr is out of range 0...issues.size()-1
	 */
	Value getCurrentEval(Bid bid, int rownr) throws Exception {
		if (bid == null)
			return null;
		Integer ID = IDs.get(rownr); // get ID of the issue in question.
		return bid.getValue(ID); // get the current value for that issue in the
									// bid
	}

	/**
	 * get a render component
	 * 
	 * @return the Component that can be rendered to show this cell.
	 */
	@Override
	public Component getValueAt(int row, int col) {
		if (row == issues.size()) {
			if (col == 0)
				return new JLabel("Utility:");
			if (utilitySpace == null)
				return new JLabel("No UtilSpace");
			Bid bid;
			if (col == 1)
				bid = lastAccepted;
			else
				try {
					bid = getBid();
				} catch (Exception e) {
					bid = null;
					System.out.println("Internal err with getBid:"
							+ e.getMessage());
				}
			;
			JProgressBar bar = new JProgressBar(0, 100);
			bar.setStringPainted(true);
			try {
				bar.setValue((int) (0.5 + 100.0 * utilitySpace.getUtility(bid)));
				bar.setIndeterminate(false);
			} catch (Exception e) {
				new Warning("Exception during cost calculation:"
						+ e.getMessage(), false, 1);
				bar.setIndeterminate(true);
			}

			return bar;
		}

		switch (col) {
		case 0:
			return new JTextArea(issues.get(row).getName());
		case 1:
			Value value = null;
			try {
				value = getCurrentEval(lastAccepted, row);
			} catch (Exception e) {
				System.out.println("Err EnterBidDialog2.getValueAt: "
						+ e.getMessage());
			}
			if (value == null)
				return new JTextArea("-");
			return new JTextArea(value.toString());
		case 2:
			return comboBoxes.get(row);
		}
		return null;
	}

	Bid getBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();

		for (int i = 0; i < issues.size(); i++)
			values.put(IDs.get(i), ((IssueDiscrete) issues.get(i))
					.getValue(comboBoxes.get(i).getSelectedIndex()));
		// values.put(IDs.get(i), (Value)comboBoxes.get(i).getSelectedItem());
		return new Bid(utilitySpace.getDomain(), values);
	}

	public void actionPerformed(ActionEvent e) {
		// System.out.println("event d!"+e);
		// receiveMessage the cost and utility of our own bid.
		fireTableCellUpdated(issues.size(), 2);
		fireTableCellUpdated(issues.size() + 1, 2);
	}

}
