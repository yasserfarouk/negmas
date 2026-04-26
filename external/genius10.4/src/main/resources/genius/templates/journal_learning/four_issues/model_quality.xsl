<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
	<xsl:output method="html"/>
	<xsl:template match="/">
		<STYLE>
		H1: {COLOR: blue FONT-FAMILY: Arial; }
		SubTotal {COLOR: green;  FONT-FAMILY: Arial}
		BODY {COLOR: blue; FONT-FAMILY: Arial; FONT-SIZE: 8pt;}
		TR.clsOdd { background-Color: beige;  }
		TR.clsEven { background-color: #cccccc; }
		</STYLE>
		
		<H2>Results Listing (in Alternating row colors) </H2>

			<table border="1">
				<tr>	
					<td rowspan="2"> Prfile B </td>
					<td rowspan="2"> Prfile A </td>
					<td colspan="2"> Utility Space </td>
					<td colspan="2"> Weights </td>
				</tr>
				<tr>
					<td> Ranking </td>
					<td> Pearsion </td>
					<td> Ranking </td>
					<td> Pearsion </td>
				</tr>

			<xsl:for-each select="/Tournament/NegotiationOutcome/additional_log">
			        <tr>
				<xsl:for-each select="../resultsOfAgent">
					<td> <xsl:value-of select="@utilspace"/></td>
				</xsl:for-each>

				<xsl:for-each select="opposition">
					<td>
						<xsl:value-of select="@ranking_distance_utility_space"/>
					</td>
					<td>
						<xsl:value-of select="@pearson_distance_utility_space"/>
					</td>
				</xsl:for-each>
				<xsl:for-each select="learning_performance">
					<td>
						<xsl:value-of select="@ranking_distance_utility_space"/>
					</td>
					<td>
						<xsl:value-of select="@pearson_distance_utility_space"/>
					</td>
					<td>
						<xsl:value-of select="@ranking_distance_weights"/>
					</td>
					<td>
						<xsl:value-of select="@pearson_distance_weights"/>
					</td>

				</xsl:for-each>
	                	</tr>
			</xsl:for-each>
			
			</table>

		<H3>Total Rounds <xsl:value-of select="count(Tournament/NegotiationOutcome)"/>
		</H3>
	</xsl:template>

</xsl:stylesheet>