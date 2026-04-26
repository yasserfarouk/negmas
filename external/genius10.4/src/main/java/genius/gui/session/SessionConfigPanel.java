package genius.gui.session;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.repository.DomainRepItem;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.gui.deadline.DeadlinePanel;
import genius.gui.panels.CheckboxPanel;
import genius.gui.panels.ComboboxSelectionPanel;
import genius.gui.panels.VflowPanelWithBorder;
import genius.gui.renderer.RepItemListCellRenderer;

/**
 * Panel that allows the user to set up the (multilateral) session settings
 */
@SuppressWarnings("serial")
public class SessionConfigPanel extends VflowPanelWithBorder {

	private SessionModel model;
	private BilateralOptionsPanel bilateralOptions;

	public SessionConfigPanel(SessionModel model) {
		super("Multiparty Negotiation Session Setup");
		this.model = model;

		initPanel();
	}

	/**
	 * Load and set all the panel elements - buttons, comboboxes, etc.
	 */
	private void initPanel() {
		ComboboxSelectionPanel<MultiPartyProtocolRepItem> protocolcomb = new ComboboxSelectionPanel<>("Protocol",
				model.getProtocolModel());
		protocolcomb.setCellRenderer(new RepItemListCellRenderer());
		
		ComboboxSelectionPanel<DomainRepItem> domaincomb = new ComboboxSelectionPanel<>("Domain",
				model.getDomainModel());

		MediatorPanel mediatorpanel = new MediatorPanel(model.getMediatorIdModel(), model.getMediatorModel());

		add(protocolcomb);
		add(domaincomb);
		add(mediatorpanel);

		add(new ParticipantsPanel(model.getParticipantModel(), model.getParticipantsModel()));
		add(new DeadlinePanel(model.getDeadlineModel()));
		add(new ComboboxSelectionPanel<>("Data persistency", model.getPersistentDatatypeModel()));
		add(new CheckboxPanel("Enable System.out print", model.getPrintEnabled()));
		add(new CheckboxPanel("Enable progress graph", model.getShowChart()));

		bilateralOptions = new BilateralOptionsPanel(model.getBilateralUtilUtilPlot(), model.getBilateralShowAllBids());
		add(bilateralOptions);

		JButton start = new JButton("Start");
		add(start);
		start.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				model.modelIsComplete();
			}
		});

		model.getParticipantsModel().addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {
				updateVisibility();
			}

			@Override
			public void intervalAdded(ListDataEvent e) {
				updateVisibility();
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				updateVisibility();
			}
		});
		updateVisibility();
	}

	private void updateVisibility() {
		bilateralOptions.setVisible(model.getParticipantsModel().getSize() == 2);
	}

}
