package agents.anac.y2014.kGA_gent.library_genetic;

import java.util.List;


/*
 * ä¸–ä»£æ›´æ–°ã‚¤ãƒ³ã‚¿ãƒ¼ãƒ•ã‚§ãƒ¼ã‚¹
 * å®Ÿè£…ã�™ã‚‹ã�¹ã��ã�¯ï¼Œé�ºä¼�å­�ãƒªã‚¹ãƒˆã‚’å�—ã�‘å�–ã‚Šã€�æ¬¡ä¸–ä»£ã�®é�ºä¼�å­�ãƒªã‚¹ãƒˆã‚’è¿”å�´ã�™ã‚‹
 *
 * é�ºä¼�å­�æ›´æ–°ã�®çµ‚äº†åˆ¤æ–­
 */

public interface GenerationChange {



	List<Gene> Generation(List<Gene> list);

	/*
	 * åˆ�æœŸé�ºä¼�å­�ãƒªã‚¹ãƒˆã�®ç”Ÿæˆ�
	 */
	List<Gene> StartGeneration(Gene gene);

	List<Gene> StartGeneration();

	boolean End(List<Gene> list);

}
