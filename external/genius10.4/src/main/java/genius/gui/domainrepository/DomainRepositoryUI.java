package genius.gui.domainrepository;

import java.awt.Font;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPopupMenu;
import javax.swing.JTree;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreePath;
import javax.swing.tree.TreeSelectionModel;

import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.repository.DomainRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.RepItem;
import genius.core.repository.Repository;
import genius.core.repository.RepositoryFactory;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;
import genius.core.xml.SimpleElement;
import genius.gui.GeniusAppInterface;
import genius.gui.tree.DomainAndProfileEditorPanel;

/**
 * A user interface to the domain repository. Shows all available domains in a
 * (usually long) list.
 * 
 */
@SuppressWarnings("serial")
public class DomainRepositoryUI extends JTree {
	private Repository<DomainRepItem> domainrepository;
	private MyTreeNode root = new MyTreeNode(null);
	private DefaultTreeModel scenarioTreeModel;
	private GeniusAppInterface mainPanel;

	public DomainRepositoryUI(GeniusAppInterface negoView) {
		domainrepository = RepositoryFactory.get_domain_repos();
		initTree();
		setModel(scenarioTreeModel);
		this.mainPanel = negoView;

		setMinimumSize(new java.awt.Dimension(100, 100));
		setName("treeDomains");

		addMouseListener(new java.awt.event.MouseAdapter() {
			@Override
			public void mouseClicked(java.awt.event.MouseEvent evt) {
				treeDomainsMouseClicked(evt);
			}
		});

	}

	private void initTree() {
		// for all domains in the domain repository
		for (DomainRepItem repitem : domainrepository.getItems()) {
			DomainRepItem dri = repitem;
			MyTreeNode domainNode = new MyTreeNode(dri);
			// add all preference profiles of the domain as nodes
			for (ProfileRepItem profileitem : dri.getProfiles()) {
				domainNode.add(new MyTreeNode(profileitem));
			}
			root.add(domainNode);
		}

		scenarioTreeModel = new DefaultTreeModel(root);
		setModel(scenarioTreeModel);
		Font currentFont = getFont();
		Font bigFont = new Font(currentFont.getName(), currentFont.getStyle(),
				currentFont.getSize() + 2);
		setRowHeight(23);
		setFont(bigFont);
		setRootVisible(false);
		setShowsRootHandles(true);
		getSelectionModel()
				.setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
		addMouseListener(new MouseAdapter() {

			// for Windows
			@Override
			public void mouseReleased(MouseEvent e) {
				mouseCode(e);
			}

			// for Linux
			@Override
			public void mousePressed(MouseEvent e) {
				mouseCode(e);
			}

			private void mouseCode(MouseEvent e) {
				if (e.getButton() == MouseEvent.BUTTON3) {
					TreePath selPath = getPathForLocation(e.getX(), e.getY());
					if (selPath == null) {
						JPopupMenu popup = createPopupMenu(null);
						popup.show(e.getComponent(), e.getX(), e.getY());
						return;
					}

					MyTreeNode node = (MyTreeNode) selPath
							.getLastPathComponent();
					setSelectionPath(selPath);
					if (e.isPopupTrigger()
							&& e.getComponent() instanceof JTree) {
						JPopupMenu popup = createPopupMenu(node);
						popup.show(e.getComponent(), e.getX(), e.getY());
					}
				}
			}
		});
	}

	private JPopupMenu createPopupMenu(final MyTreeNode node) {
		JPopupMenu popup = new JPopupMenu();

		JMenuItem newDomain = new JMenuItem("New domain");
		newDomain.addActionListener(new java.awt.event.ActionListener() {
			@Override
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				addDomain();
			}
		});

		popup.add(newDomain);

		if (node != null) {
			if (node.getRepositoryItem() instanceof ProfileRepItem) {
				JMenuItem deletePP = new JMenuItem("Delete preference profile");
				deletePP.addActionListener(new java.awt.event.ActionListener() {
					@Override
					public void actionPerformed(
							java.awt.event.ActionEvent evt) {
						deleteProfile(node);
					}
				});
				popup.add(deletePP);
			} else {
				JMenuItem newPP = new JMenuItem("New preference profile");
				newPP.addActionListener(new java.awt.event.ActionListener() {
					@Override
					public void actionPerformed(
							java.awt.event.ActionEvent evt) {
						if (domainHasIssues(node)) {
							newPreferenceProfile(node);
						} else {
							showError(
									"Before creating a preference profile, the domain must be saved with at least one isue.");
						}
					}
				});
				popup.add(newPP);
				JMenuItem deleteDomain = new JMenuItem("Delete domain");
				deleteDomain
						.addActionListener(new java.awt.event.ActionListener() {
							@Override
							public void actionPerformed(
									java.awt.event.ActionEvent evt) {
								deleteDomain(node);
							}
						});
				popup.add(deleteDomain);
			}
		}
		return popup;
	}

	protected boolean domainHasIssues(MyTreeNode node) {
		// get the directory of the domain
		DomainRepItem dri = (DomainRepItem) node.getRepositoryItem();
		String fullPath = dri.getURL().toString().substring(5);
		Domain domain = null;
		try {
			domain = new DomainImpl(fullPath);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return domain != null && domain.getIssues().size() > 0;
	}

	private void newPreferenceProfile(MyTreeNode node) {

		DomainRepItem dri = (DomainRepItem) node.getRepositoryItem();
		String domainName = dri.getURL().toString().replace("file:", "");
		String cleanName = domainName.replace(".xml", "").replace("_domain",
				"");
		String path = "";

		int i = 1;
		boolean found = false;
		while (!found && i < 100) {
			path = cleanName + "_util" + i + ".xml";
			File file = new File(path);
			if (!file.exists()) {
				found = true;
			}
			i++;
		}
		String url = "file:" + path;
		ProfileRepItem newPref = null;

		try {
			newPref = new ProfileRepItem(new URL(url), dri);
			dri.getProfiles().add(newPref);
			MyTreeNode newNode = new MyTreeNode(newPref);
			scenarioTreeModel.insertNodeInto(newNode, node,
					node.getChildCount());
			domainrepository.save();

			Domain domain = null;
			try {
				domain = new DomainImpl(domainName);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			AdditiveUtilitySpace space = null;
			try {
				space = new AdditiveUtilitySpace(domain, "");
			} catch (Exception e) {
				e.printStackTrace();
			}
			try {
				space.toXML().saveToFile(path);
			} catch (IOException e) {
				e.printStackTrace();
			}
			showRepositoryItemInTab(newPref, newNode);
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}
	}

	private void addDomain() {

		String result = new CreateNewDomain(mainPanel.getMainFrame())
				.getResult();
		if (result != null) {
			String dirName = "etc" + File.separator + "templates"
					+ File.separator + result;

			File dir = new File(dirName);
			if (!dir.exists()) {
				dir.mkdir();
			}

			String path = dirName + File.separator + result + ".xml";
			DomainRepItem dri = null;
			try {
				dri = new DomainRepItem(new URL("file:" + path));
			} catch (MalformedURLException e) {
				e.printStackTrace();
			}
			domainrepository.getItems().add(dri);
			domainrepository.save();
			MyTreeNode newNode = new MyTreeNode(dri);
			scenarioTreeModel.insertNodeInto(newNode, root,
					root.getChildCount());
			saveDomainAsFile(path, result);
			showRepositoryItemInTab(dri, newNode);
			updateUI();
		}
	}

	private void saveDomainAsFile(String relativePath, String domainName) {
		SimpleElement template = new SimpleElement("negotiation_template");
		SimpleElement utilSpace = new SimpleElement("utility_space");
		SimpleElement objective = new SimpleElement("objective");
		objective.setAttribute("index", "0");
		objective.setAttribute("description", "");
		objective.setAttribute("name", domainName);
		objective.setAttribute("type", "objective");
		objective.setAttribute("etype", "objective");
		utilSpace.addChildElement(objective);
		template.addChildElement(utilSpace);
		template.saveToFile(relativePath);
	}

	private void deleteDomain(MyTreeNode node) {
		DomainRepItem dri = (DomainRepItem) node.getRepositoryItem();
		scenarioTreeModel.removeNodeFromParent(node);
		domainrepository.getItems().remove(dri);
		domainrepository.save();
	}

	private void deleteProfile(MyTreeNode node) {
		ProfileRepItem pri = (ProfileRepItem) node.getRepositoryItem();
		scenarioTreeModel.removeNodeFromParent(node);
		domainrepository.removeProfileRepItem(pri);
		domainrepository.save();
	}

	private void treeDomainsMouseClicked(java.awt.event.MouseEvent evt) {
		int selRow = getRowForLocation(evt.getX(), evt.getY());
		TreePath selPath = getPathForLocation(evt.getX(), evt.getY());
		if (selRow != -1) {
			if (evt.getClickCount() == 2) {
				if (selPath != null) {
					MyTreeNode node = (MyTreeNode) (selPath
							.getLastPathComponent());
					RepItem repItem = node.getRepositoryItem();
					showRepositoryItemInTab(repItem, node);
				}
			}
		}
	}

	public void showRepositoryItemInTab(RepItem repItem, MyTreeNode node) {
		DomainAndProfileEditorPanel tf;
		if (repItem instanceof DomainRepItem) {
			try {
				boolean hasNoProfiles = ((DomainRepItem) repItem).getProfiles()
						.size() == 0;
				String filename = ((DomainRepItem) repItem).getURL().getFile();
				DomainImpl domain = new DomainImpl(filename);
				tf = new DomainAndProfileEditorPanel(domain, hasNoProfiles);
				mainPanel.addTab(StripExtension(GetPlainFileName(filename)),
						tf);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (repItem instanceof ProfileRepItem) {
			try {
				MyTreeNode parentNode = (MyTreeNode) (node.getParent());
				String filename = ((ProfileRepItem) repItem).getURL().getFile();

				DomainImpl domain = new DomainImpl(
						((DomainRepItem) (parentNode.getRepositoryItem()))
								.getURL().getFile());

				// we can handle only AdditiveUtilitySpace in TreeFrame anyway
				// so we guess it's that...
				UtilitySpace profile = ((ProfileRepItem) repItem).create();
				if (!(profile instanceof AdditiveUtilitySpace)) {
					showError(
							"Selected profile is not an Additive Linear profile, you can not edit this.");
					return;
				}
				AdditiveUtilitySpace utilitySpace = (AdditiveUtilitySpace) profile;

				tf = new DomainAndProfileEditorPanel(domain, utilitySpace);
				mainPanel.addTab(StripExtension(GetPlainFileName(filename)),
						tf);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	private void showError(String string) {
		JOptionPane.showMessageDialog(null, string, "Domain error", 0);
	}

	/**
	 * @param filename
	 * @return part of filename following the last slash, or full filename if
	 *         there is no slash.
	 */
	public String GetPlainFileName(String filename) {
		int i = filename.lastIndexOf('/');
		if (i == -1)
			return filename;
		return filename.substring(i + 1);
	}

	/**
	 * @param filename
	 * @return filename stripped of its extension (the part after the last dot).
	 */
	public String StripExtension(String filename) {
		int i = filename.lastIndexOf('.');
		if (i == -1)
			return filename;
		return filename.substring(0, i);
	}

}