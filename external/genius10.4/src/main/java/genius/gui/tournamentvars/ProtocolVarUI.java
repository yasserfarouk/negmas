package genius.gui.tournamentvars;

import java.awt.Frame;
import java.awt.Panel;
import java.util.ArrayList;

import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JRadioButton;

import genius.core.repository.ProtocolRepItem;
import genius.core.repository.Repository;
import genius.core.repository.RepositoryFactory;
import genius.gui.panels.DefaultOKCancelDialog;

/**
 * Open a UI and negotiate with user about which agents to use in tournament.
 * 
 * @author wouter
 * 
 */
public class ProtocolVarUI extends DefaultOKCancelDialog {

	private static final long serialVersionUID = -6106919299675060907L;
	ArrayList<ProtocolRadioButton> radioButtons; // copy of what's in the panel,
													// for easy check-out.

	public ProtocolVarUI(Frame owner) {
		super(owner, "Protocol Variable Selector");

	}

	public Panel getPanel() {
		radioButtons = new ArrayList<ProtocolRadioButton>();
		Panel protocolList = new Panel();
		protocolList.setLayout(new BoxLayout(protocolList, BoxLayout.Y_AXIS));
		ButtonGroup group = new ButtonGroup();

		Repository<ProtocolRepItem> protocolRep = RepositoryFactory.getProtocolRepository();
		for (ProtocolRepItem agt : protocolRep.getItems()) {
			ProtocolRadioButton cbox = new ProtocolRadioButton((ProtocolRepItem) agt);
			radioButtons.add(cbox);
			protocolList.add(cbox);
			cbox.setSelected(true);
			group.add(cbox);
		}
		return protocolList;
	}

	public Object ok() {
		ArrayList<ProtocolRepItem> result = new ArrayList<ProtocolRepItem>();
		for (ProtocolRadioButton cbox : radioButtons) {
			if (cbox.isSelected())
				result.add(cbox.protocolRepItem);
		}
		return result;
	}
}

class ProtocolRadioButton extends JRadioButton {
	public ProtocolRepItem protocolRepItem;

	public ProtocolRadioButton(ProtocolRepItem protocolRepItem) {
		super("" + protocolRepItem.getName());
		this.protocolRepItem = protocolRepItem;
	}
}