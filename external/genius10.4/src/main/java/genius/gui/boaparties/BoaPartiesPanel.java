package genius.gui.boaparties;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.AbstractAction;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import genius.core.exceptions.InstantiateException;
import genius.core.repository.BoaPartyRepository;
import genius.core.repository.RepositoryFactory;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * A panel that shows available BOA parties and allows you to add/remove
 *
 */
@SuppressWarnings("serial")
public class BoaPartiesPanel extends JPanel {

	public BoaPartiesPanel() {
		setLayout(new BorderLayout());
		JTable table = new JTable(new BoaPartiesModel());
		JScrollPane scrollpane = new JScrollPane(table);
		add(scrollpane, BorderLayout.CENTER);

		PopClickListener popuplistener = new PopClickListener(table);
		scrollpane.addMouseListener(popuplistener);
		table.addMouseListener(popuplistener);
	}

	public static void main(String[] args) {
		JFrame frame = new JFrame();
		frame.setLayout(new BorderLayout());
		frame.getContentPane().add(new BoaPartiesPanel(), BorderLayout.CENTER);
		frame.pack();
		frame.setVisible(true);
	}
}

/**
 * Right mouse click menu.
 *
 */
class PopClickListener extends MouseAdapter {
	private JTable table;

	public PopClickListener(JTable table) {
		this.table = table;
	}

	public void mousePressed(MouseEvent e) {
		if (e.isPopupTrigger())
			doPop(e);
	}

	public void mouseReleased(MouseEvent e) {
		if (e.isPopupTrigger())
			doPop(e);
	}

	private void doPop(MouseEvent e) {
		PopUp menu = new PopUp(table);
		menu.show(e.getComponent(), e.getX(), e.getY());
	}
}

@SuppressWarnings("serial")
class PopUp extends JPopupMenu {

	public PopUp(JTable table) {
		add(new JMenuItem(new AddAction(table)));
		add(new JMenuItem(new EditAction(table)));
		add(new JMenuItem(new RemoveAction(table)));
	}
}

/**
 * Action to add an item to the boaparty repo
 *
 */
@SuppressWarnings("serial")
class AddAction extends AbstractAction {

	private BoaPartyRepository partymodel = RepositoryFactory.getBoaPartyRepository();

	public AddAction(JTable table) {
		super("add item");
		putValue(SHORT_DESCRIPTION, "add a new boa item");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		try {
			BoaPartyModel model = new BoaPartyModel(new BoaPartyRepItem("newParty"));
			final BoaPartyPanel panel = new BoaPartyPanel(model);
			int result = JOptionPane.showConfirmDialog(null, panel, "Add new party", JOptionPane.OK_CANCEL_OPTION);
			if (result == JOptionPane.OK_OPTION) {
				System.out.println("ADDING " + model.getValues());
				partymodel.addAll(model.getValues());
			}
		} catch (InstantiateException e2) {
			e2.printStackTrace();
		}
	}
}

/**
 * Action to remove an item to the boaparty repo
 *
 */
@SuppressWarnings("serial")
class RemoveAction extends AbstractAction {

	private BoaPartyRepository partymodel = RepositoryFactory.getBoaPartyRepository();
	private JTable table;

	public RemoveAction(JTable table) {
		super("remove item");
		this.table = table;
		putValue(SHORT_DESCRIPTION, "remove a boa item");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		int selectedrow = table.getSelectedRow();
		if (selectedrow == -1)
			return;
		BoaPartyRepItem selectedparty = partymodel.getList().getList().get(selectedrow);
		partymodel.remove(selectedparty);
	}

}

/**
 * Action to edit an item to the boaparty repo
 *
 */
@SuppressWarnings("serial")
class EditAction extends AbstractAction {

	private BoaPartyRepository partymodel = RepositoryFactory.getBoaPartyRepository();
	private JTable table;

	public EditAction(JTable table) {
		super("edit item");
		this.table = table;
		putValue(SHORT_DESCRIPTION, "edit a boa item");
	}

	@Override
	public void actionPerformed(ActionEvent e) {

		try {
			int selectedrow = table.getSelectedRow();
			if (selectedrow == -1)
				return;
			BoaPartyRepItem selectedparty = partymodel.getList().getList().get(selectedrow);
			BoaPartyModel model = new BoaPartyModel(selectedparty);
			final BoaPartyPanel panel = new BoaPartyPanel(model);

			int result = JOptionPane.showConfirmDialog(null, panel, "Edit party", JOptionPane.OK_CANCEL_OPTION);
			if (result == JOptionPane.OK_OPTION) {
				partymodel.remove(selectedparty);
				partymodel.addAll(model.getValues());
			}
		} catch (InstantiateException e1) {
			e1.printStackTrace();
		}

	}

}
