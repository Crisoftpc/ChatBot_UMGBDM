<?php
/*
Plugin name: ChatCompose Chatbot
Plugin URI: https://www.chatcompose.com
Description: A wordpress plugin to install a ChatCompose chatbot
Author: ChatCompose
Author URI: https://www.chatcompose.com
Version: 0.1
*/

// Call chatcompose_id_menu function to load plugin menu in dashboard
add_action( 'admin_menu', 'chatcompose_id_menu' );

// Create WordPress admin menu
if( !function_exists("chatcompose_id_menu") )
{
function chatcompose_id_menu(){

  $page_title = 'ChatCompose Chatbot';
  $menu_title = 'ChatCompose';
  $capability = 'manage_options';
  $menu_slug  = 'extra-post-info';
  $function   = 'chatcompose_id_page';
  $icon_url   = 'dashicons-media-code';
  $position   = 4;

  add_menu_page( $page_title,
                 $menu_title,
                 $capability,
                 $menu_slug,
                 $function,
                 $icon_url,
                 $position );

  // Call update_chatcompose_id function to update database
  add_action( 'admin_init', 'update_chatcompose_id' );

}
}

// Create function to register plugin settings in the database
if( !function_exists("update_chatcompose_id") )
{
function update_chatcompose_id() {
  register_setting( 'chatcompose_id-settings', 'chatcompose_id' );
  register_setting( 'chatcompose_id-settings', 'lang_id' );
}
}

// Create WordPress plugin page
if( !function_exists("chatcompose_id_page") )
{
  
function chatcompose_id_page(){
?>
  <h1>ChatCompose Chatbot</h1>
  <form method="post" action="options.php">
    <?php settings_fields( 'chatcompose_id-settings' ); ?>
    <?php do_settings_sections( 'chatcompose_id-settings' ); ?>
    <table class="form-table">
      <tr valign="top">
      <th scope="row">ChatCompose user id:</th>
      <td><input type="text" name="chatcompose_id" value="<?php echo get_option('chatcompose_id'); ?>"/></td>
      </tr>
      <tr valign="top">
      <th scope="row">Language Code (Ex: EN, DE, RU):</th>
      <td><input type="text" name="lang_id" value="<?php echo get_option('lang_id'); ?>"/></td>
      </tr>
    </table>
    <p></p>
    <p>Don't have an account? </p> <p><a href="https://admin.chatcompose.com/login">Register for free</a></p>
    <p>Don't know your language code? </p> <p><a href="https://www.sitepoint.com/iso-2-letter-language-codes/">Check it here</a></p>
  <?php submit_button(); ?>
  </form>
<?php

}
}

add_action( 'wp_head', 'tws_output_meta_tags', 1);

if( !function_exists("tws_output_meta_tags") )
{
function tws_output_meta_tags() {
    $extra_info = get_option('chatcompose_id');
    $extra_lang = get_option('lang_id');
    /*
    SAME OUTPUT
    echo "<link rel='apple-touch-icon-precomposed' sizes='57x57' href='${base_url_favicon}apple-touch-icon-57x57.png' />\n";
    echo '<link rel="apple-touch-icon-precomposed" sizes="57x57" href="'.${base_url_favicon}.'apple-touch-icon-57x57.png" />'."\n"; 
    */
    if($extra_info && $extra_lang)
        {
    echo '<link href="https://chatcompose.azureedge.net/static/all/global/export/css/main.5b1bd1fd.css" rel="stylesheet">'."\n";
    echo '<script async type="text/javascript" src="https://chatcompose.azureedge.net/static/all/global/export/js/main.a7059cb5.js" user="'.$extra_info.'" lang="'.$extra_lang.'"></script>'."\n";
}    
}
} 
?>