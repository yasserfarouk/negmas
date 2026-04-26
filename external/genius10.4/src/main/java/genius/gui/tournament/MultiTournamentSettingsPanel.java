package genius.gui.tournament;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;

import genius.core.config.MultilateralTournamentConfiguration;
import genius.core.listener.Listener;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.PartyRepItem;
import genius.gui.deadline.DeadlinePanel;
import genius.gui.panels.CheckboxPanel;
import genius.gui.panels.ComboboxSelectionPanel;
import genius.gui.panels.SpinnerPanel;
import genius.gui.panels.VflowPanelWithBorder;
import genius.gui.renderer.RepItemListCellRenderer;

/**
 * This is the user interface for the multilateral tournament and replaces the
 * old MultiTournamentUI.
 * <p/>
 * The configuration of this user interface is stored in the
 * {@link MultilateralTournamentConfiguration} variable, which is also used by
 * the tournament manager to run the tournaments.
 *
 * @author W.Pasman
 *
 */
@SuppressWarnings("serial")
public class MultiTournamentSettingsPanel extends VflowPanelWithBorder {

	private MultiTournamentModel model;
	private JButton start = new JButton("Start Tournament");
	private BilateralOptionsPanel biOptionsPanel;

	public MultiTournamentSettingsPanel(MultiTournamentModel model) {
		super("Multilateral negotiation Tournament Setup");
		this.model = model;
		initPanel();
	}

	/**
	 * Load and set all the panel elements - buttons, comboboxes, etc.
	 */
	private void initPanel() {
		ComboboxSelectionPanel<MultiPartyProtocolRepItem> protocolcomb = new ComboboxSelectionPanel<>(
				"Protocol", model.getProtocolModel());
		protocolcomb.setCellRenderer(new RepItemListCellRenderer());
		ComboboxSelectionPanel<PartyRepItem> mediatorcomb = new ComboboxSelectionPanel<>(
				"Mediator", model.getMediatorModel());
		mediatorcomb.setCellRenderer(new RepItemListCellRenderer());

		add(protocolcomb);
		add(new DeadlinePanel(model.getDeadlineModel()));
		add(new SpinnerPanel("Nr. Tournaments",
				model.getNumTournamentsModel()));
		add(new SpinnerPanel("Agents per Session",
				model.getNumAgentsPerSessionModel()));
		add(new CheckboxPanel("Agent Repetition",
				model.getAgentRepetitionModel()));
		add(new CheckboxPanel("Randomize session order",
				model.getRandomSessionOrderModel()));
		add(new CheckboxPanel("Enable System.out print",
				model.getEnablePrint()));
		add(new ComboboxSelectionPanel<>("Data persistency",
				model.getPersistentDatatypeModel()));

		add(mediatorcomb);

		add(new PartiesAndProfilesPanel(model.getPartyModel(),
				model.getProfileModel()));

		biOptionsPanel = new BilateralOptionsPanel(
				model.getBilateralOptionsModel());
		add(biOptionsPanel);

		add(start);
		start.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				model.modelIsComplete();
			}
		});

		model.getNumAgentsPerSessionModel().addListener(new Listener<Number>() {
			@Override
			public void notifyChange(Number e) {
				updateBipanelVisibility();
			}
		});
		updateBipanelVisibility();

	}

	private void updateBipanelVisibility() {
		biOptionsPanel.setVisible(
				model.getNumAgentsPerSessionModel().getValue().intValue() == 2);
	}

	/**
	 * simple stub to run this stand-alone (for testing).
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		final JFrame gui = new JFrame();
		gui.setLayout(new BorderLayout());
		MultiTournamentModel model = new MultiTournamentModel();
		gui.getContentPane().add(new MultiTournamentSettingsPanel(model),
				BorderLayout.CENTER);
		gui.pack();
		gui.setVisible(true);

		model.addListener(new Listener<MultilateralTournamentConfiguration>() {

			@Override
			public void notifyChange(MultilateralTournamentConfiguration data) {
				System.out.println("done, with " + data);
				gui.setVisible(false);
			}
		});
	}
}
