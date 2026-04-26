package genius.gui.agentrepository;

import java.awt.Component;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;

import javax.swing.JFileChooser;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.filechooser.FileFilter;
import javax.swing.table.AbstractTableModel;

import genius.core.Global;
import genius.core.repository.AgentRepItem;
import genius.core.repository.Repository;
import genius.core.repository.RepositoryFactory;
import genius.gui.panels.GenericFileFilter;

/**
 * A user interface to the agent repository
 * 
 * @author Wouter Pasman, Mark Hendrikx
 */
public class AgentRepositoryUI {
	private static final String ADD_AN_AGENT = "Add an agent";
	private Repository<AgentRepItem> agentrepository;
	private AbstractTableModel dataModel;
	private final JTable table;

	/**
	 * appends the UI to the given scrollpane. Kind of hack, NegoGUIView has
	 * already created the scrollpane. #858 we need the scrollpane because we
	 * want to attach mouse listener to the scrollpane
	 * 
	 * @param jScrollPane1
	 */
	public AgentRepositoryUI(JScrollPane jScrollPane1) {
		this.table = new JTable();
		jScrollPane1.setViewportView(table);

		agentrepository = RepositoryFactory.get_agent_repository();

		initTable();
		addPopupMenu(jScrollPane1);
		addPopupMenu(table);

	}

	/**
	 * User clicked in our panel. Show popup menu
	 * 
	 * if there is a row selected in the table, we also enable the Remove
	 * option.
	 * 
	 * @return popupmenu
	 */
	private JPopupMenu createPopupMenu() {
		JPopupMenu popup = new JPopupMenu();

		JMenuItem addAgent = new JMenuItem("Add new agent");
		addAgent.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				addAction();
			}
		});
		popup.add(addAgent);

		if (table.getSelectedRow() > 0) {

			JMenuItem removeAgent = new JMenuItem("Remove agent");
			removeAgent.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent evt) {
					removeAction();
				}
			});
			popup.add(removeAgent);
		}

		return popup;
	}

	/**
	 * Add a popup menu to a component. We need to attach this to multiple
	 * components: the scrollpane and the table. This is because we want the
	 * menu also to appear if users click outside the table #858
	 * 
	 * @param component
	 *            the component to add the menu to.
	 * 
	 */
	private void addPopupMenu(Component component) {
		component.addMouseListener(new MouseAdapter() {

			// if Windows
			@Override
			public void mouseReleased(MouseEvent e) {
				mouseCode(e);
			}

			// if Linux
			public void mousePressed(MouseEvent e) {
				mouseCode(e);
			}

			private void mouseCode(MouseEvent e) {
				int r = table.rowAtPoint(e.getPoint());
				if (r >= 0 && r < table.getRowCount()) {
					table.setRowSelectionInterval(r, r);
				} else {
					table.clearSelection();
				}

				if (e.isPopupTrigger()) {// && e.getComponent() instanceof
											// JTable) {
					// if rowindex>0, we actually selected a row.
					JPopupMenu popup = createPopupMenu();
					popup.show(e.getComponent(), e.getX(), e.getY());
				}
			}
		});
	}

	private void initTable() {
		dataModel = new AbstractTableModel() {
			private static final long serialVersionUID = -4985008096999143587L;
			final String columnnames[] = { "Agent Name", "Description" };

			public int getColumnCount() {
				return columnnames.length;
			}

			public int getRowCount() {
				return agentrepository.getItems().size();
			}

			public Object getValueAt(int row, int col) {
				AgentRepItem agt = (AgentRepItem) agentrepository.getItems().get(row);
				switch (col) {
				case 0:
					String error = "";
					if (agt.getVersion().equals("ERR") && !agt.getName().equals(ADD_AN_AGENT)) {
						error = " (LOADING FAILED)";
					}
					return agt.getName() + error;
				case 1:
					return agt.getDescription();
				}
				return col;
			}

			public String getColumnName(int column) {
				return columnnames[column];
			}
		};

		if (agentrepository.getItems().size() == 0) {
			addTemporaryAgent();
			agentrepository.save();
		}

		table.setModel(dataModel);
		table.setShowVerticalLines(false);

		table.addKeyListener(new KeyAdapter() {
			public void keyReleased(KeyEvent ke) {
				if (ke.getKeyCode() == KeyEvent.VK_DELETE) {
					removeAction();
				}
			}
		});
	}

	/**
	 * Add new agent to repository. The agent is expected to be a .class file
	 */
	public void addAction() {

		// Restrict file picker to root and subdirectories.
		// Ok, you can escape if you put in a path as directory. We catch this
		// later on.
		//
		// FileSystemView fsv = new DirectoryRestrictedFileSystemView(new
		// File(root));
		// JFileChooser fc = new JFileChooser(fsv.getHomeDirectory(), fsv);
		// Lifted the restriction. #856
		JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));

		// Filter such that only directories and .class files are shown.
		FileFilter filter = new GenericFileFilter("class", "Java class files (.class)");
		fc.setFileFilter(filter);

		// Open the file picker
		int returnVal = fc.showOpenDialog(null);

		// If file selected
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			try {
				addToRepo(new AgentRepItem(fc.getSelectedFile()));
			} catch (Throwable e) {
				Global.showLoadError(fc.getSelectedFile(), e);
			}
		}

	}

	/**
	 * 
	 * @param file
	 *            absolute file path, eg
	 *            /Volumes/documents/NegoWorkspace3/NegotiatorGUI
	 *            /bin/agents/BayesianAgent.class TODO I think this should be
	 *            path to the root directory for the class path, eg
	 *            '/Volumes/documents/NegoWorkspace3/NegotiatorGUI/bin'
	 * @param className
	 *            the class path name, eg "agents.BayesianAgent"
	 */
	private void addToRepo(File file, String className) {
		// Remove "Add agents" if there were no agents first
		int row = table.getSelectedRow();
		if (agentrepository.getItems().get(row).getName().equals(ADD_AN_AGENT)) {
			agentrepository.getItems().remove(row);
		}

		// Load the agent and save it in the XML. -6 strips the '.class'.
		AgentRepItem rep = new AgentRepItem(file.getName().substring(0, file.getName().length() - 6), className, "");
		agentrepository.getItems().add(rep);
		agentrepository.save();
		dataModel.fireTableDataChanged();

	}

	/**
	 * Add a Agent reference to the repo.
	 * 
	 * @param agentref
	 */
	private void addToRepo(AgentRepItem agentref) {
		// Remove "Add agents" if there were no agents first
		if (agentrepository.getItems().get(0).getName().equals(ADD_AN_AGENT)) {
			agentrepository.getItems().remove(0);
		}

		agentrepository.getItems().add(agentref);
		agentrepository.save();
		dataModel.fireTableDataChanged();

	}

	public void removeAction() {
		for (int i = 0; i < table.getSelectedRows().length; i++) {
			agentrepository.getItems().remove(table.getSelectedRows()[i]);
		}
		if (dataModel.getRowCount() == 0) {
			addTemporaryAgent();
		}
		dataModel.fireTableDataChanged();
		agentrepository.save();
	}

	private void addTemporaryAgent() {
		if (dataModel.getRowCount() == 0) {
			agentrepository.getItems().add(new AgentRepItem(ADD_AN_AGENT, "", ""));
		}
	}
}